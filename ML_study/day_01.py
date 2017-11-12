from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import jieba
import numpy as np
# 特征抽取

# # 导入包
# from sklearn.feature_extraction.text import CountVectorizer
#
# # 实例化CountVectorizer
#
# vector = CountVectorizer()
#
# # 调用fit_transform输入并转换数据
#
# res = vector.fit_transform(["life is short,i like python","life is too long,i dislike python"])
#
# # 打印结果
# print(vector.get_feature_names())
#
# print(res.toarray())

# def dictvec():
#     """
#     字典数据抽取
#     :return: None
#     """
#     dict = DictVectorizer(sparse=False)
#
#     data = dict.fit_transform([{'city': '北京', 'temperature':100}, {'city': '上海', 'temperature':60}, {'city': '深圳', 'temperature':30}])
#
#     print(dict.get_feature_names())
#
#     print(data)
#
#     return None


def cutword():
    # jieba分词
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    content1 = []
    content2 = []
    content3 = []

    # 循环取出分词的结果，放入列表，返回空格隔开的字符串
    for word in con1:
        content1.append(word)
    for word in con2:
        content2.append(word)
    for word in con3:
        content3.append(word)

    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1, c2, c3



#
# def countvec():
#     """
#     文本的特征抽取
#     :return: None
#     """
#     # 分词
#     c1, c2, c3 = cutword()
#
#     cv = CountVectorizer()
#
#     data = cv.fit_transform([c1, c2, c3])
#
#     print(cv.get_feature_names())
#     print(data.toarray())
#     return None


# def tfidfvec():
#     """
#     文本的特征抽取
#     :return: None
#     """
#     # 分词
#     c1, c2, c3 = cutword()
#
#     tf = TfidfVectorizer(stop_words=["所以", "明天"])
#
#     data = tf.fit_transform([c1, c2, c3])
#
#     print(tf.get_feature_names())
#     print(data.toarray())
#     return None

#
# def minmax():
#     """
#     特征归一化
#     :return: None
#     """
#     mm = MinMaxScaler(feature_range=(2, 3))
#
#     data = mm.fit_transform([[90,2,10,40],[60,4,15,45],[75,3,13,46]])
#
#     print(data)
#
#     return None


# def stand():
#     """
#     特征标准化
#     :return: None
#     """
#     std = StandardScaler()
#
#     data = std.fit_transform([[ 1., -1., 3.],[ 2., 4., 2.],[ 4., 6., -1.]])
#
#     print(data)
#     return None


# def im():
#     """
#     处理缺失值
#     :return: None
#     """
#     imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#
#     data = imputer.fit_transform([[1, 2], [np.nan, 3], [7, 6]])
#
#     print(data)
#     return None


# def variance():
#     """
#     删除低方差特征
#     :return: None
#     """
#     vt = VarianceThreshold()
#
#     data = vt.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
#
#     print(data)
#     return None


#
# def pca():
#     """
#     PCA数据集降维
#     :return:
#     """
#     pc = PCA(n_components=3)
#
#     data = pc.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])
#
#     print(data)
#     return None

#
# if __name__ == "__main__":
#     pca()




























