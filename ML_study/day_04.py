import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# 1、图

# g = tf.Graph()
#
# print(g)
#
# with g.as_default():
#     con = tf.constant(10.0)
#     print(con.graph)
#
# # 并不是所有的定义的东西称之为变量
# con_1 = tf.constant(3.0)
#
# con_2 = tf.constant(4.0)
#
# print(con_1, con_2)
#
# sum = tf.add(con_1, con_2)
#
# # print(sum)
#
# # 图就是一个资源
# print(tf.get_default_graph())
#
# # 会话
# with tf.Session() as sess:
#     print(con_1.graph)
#     print(sum.graph)
#     print(sess.graph)
#
#     # print(sess.run([con_1, sum]))
#
#     # 不能使用会话所在图的其它图的资源
#     print(sess.run(con))



# 2、会话

# 平常定义的Python这些变量，在会话当中是不能运行
# feed_dict机制，提供实时覆盖数据，结合placeholder使用
# 重载的机制,提供给Python当中类型与TensorFlow当中的类型结合运算，运算符背重载

# a = tf.constant(11.0)
#
# var1 = 0
# var2 = 1
#
# # x = var1 + var2
#
# # print(x)
#
# x = var1 + a
#
# print(x)
#
# # sess = tf.Session()
# #
# # print(sess.run(a))
# #
# # sess.close()
#
# con_1 = tf.constant(3.0)
#
# con_2 = tf.constant(4.0)
#
# print(con_1, con_2)
#
# sum_add = tf.add(con_1, con_2)
#
# # 占位符
# plt = tf.placeholder(tf.float32, [None, 3])
#
#
# with tf.Session() as sess:
#     # print(sess.run(sum_add))
#
#     print(sess.run(plt, feed_dict={plt: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}))
#
#     print(sess.run(x))
#     # print(con_1.eval())



# 3、张量
# 数据：Tensor  其它的都基本是：  operation
#
# con = tf.constant(5.0)
#
# print(con)
#
# with tf.Session() as sess:
#     print("---------")
#     print(con.graph)
#     print("---------")
#     print(con.op)
#     print("---------")
#     print(con.name)
#     print("---------")
#     print(con.shape)

# 静态形状和动态形状（重点）

# 对于静态形状不能去跨阶数转换
# 如果初始状态未知，可以设置静态形状, 如果固定了形状，则不能修改
# 对于想要修改形状，使用动态形状的时候，注意张量里的元素数量必须要一样

# con = tf.constant([1, 2, 3, 4])
# con.set_shape([2, 2])# 不能进行转换
#
# plt = tf.placeholder(tf.float32, [None, 3])
#
# print(plt.get_shape())
#
# plt.set_shape([2, 3])
#
# print(plt.get_shape())
#
# # plt.set_shape([3, 3])# 不能再次设置
#
# # new_tensor = tf.reshape(plt, [3, 3]) # 错误
#
# print(new_tensor.get_shape())
#
# # 对于静态形状
#
# with tf.Session() as sess:
#     # print(con.shape)
#     pass


# 变量
# 1、值也是一种张量
# 2、需要手动初始化，才能使用（注意）
# 3、能够进行显示存储
# 4、显示

# con = tf.constant(3.0)
#
# sum_add = tf.add(con, 1.0)
#
# var = tf.Variable([[1, 2], [3, 4]], name="x_data")
#
# var1 = tf.Variable([[1, 42], [3, 4]], name="x_data")
#
# print(con, var)
#
# # 初始化
# init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     # 初始化变量
#     sess.run(init_op)
#
#     tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)
#
#     print(sess.run([con, var]))


# 自实现线性回归
# def mylinearregression():
#     """
#     自实现线性回归 y = xw + b
#     :return: None
#     """
#     with tf.variable_scope("data"):
#         # 1、准备数据，x  matrix [100, 1]    y = [100]
#         X = tf.Variable(tf.random_normal([100, 1], mean=0.0, stddev=1.0), name="x_train")
#
#         # 真实目标值
#         y_true = tf.matmul(X, tf.constant([[0.7]])) + 0.8
#
#     with tf.variable_scope("model"):
#         # 2、建立模型
#         # 随机初始化权重，偏置,要保存模型，必须使用变量
#         weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name="weights")
#         bias = tf.Variable(0.0)
#
#         y_predict = tf.matmul(X, weight) + bias
#
#     with tf.variable_scope("optimizer"):
#         # 3、计算均方误差（损失）
#         loss = tf.reduce_mean(tf.square(y_true - y_predict))
#
#         # 4、优化损失值,学习率 小于1的值
#         train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#
#     # 收集变量
#     tf.summary.scalar("loss", loss)
#
#     tf.summary.scalar("biases", bias)
#
#     tf.summary.histogram("weight", weight)
#
#     # 变量初始化
#     init_op = tf.global_variables_initializer()
#
#     # 合并变量op
#     merged = tf.summary.merge_all()
#
#     # 创建一个保存实例
#     saver = tf.train.Saver()
#
#     # 开启会话运行程序
#     with tf.Session() as sess:
#         # 初始化变量
#         sess.run(init_op)
#
#         print("初始化的权重: %f和偏置: %f" % (weight.eval(), bias.eval()))
#
#         # 构造事件文件
#         filewriter = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)
#
#         # 加载模型
#         if os.path.exists("./tmp/ckpt/model/checkpoint"):
#             saver.restore(sess, "./tmp/ckpt/model/testlinear")
#
#         # 指定步数，去运行优化器
#         for i in range(10):
#             # 运行train_op
#             sess.run(train_op)
#
#             print("第%d次训练的权重: %f和偏置: %f" % (i, weight.eval(), bias.eval()))
#
#             # 运行合并变量，写入文件
#             summ = sess.run(merged)
#
#             filewriter.add_summary(summ, i)
#
#         saver.save(sess, "./tmp/ckpt/model/testlinear")
#
#     return None
#
#
# if __name__ == "__main__":
#     mylinearregression()

tf.app.flags.DEFINE_string("data_dir", "./tmp/", "数据加载的目录")
tf.app.flags.DEFINE_integer("train_step", 1000, "训练的步数")

FLAGS = tf.app.flags.FLAGS


# 命令行参数
def main(argv):
    print(argv)

    print(FLAGS.data_dir)

    print(FLAGS.train_step)


if __name__ == "__main__":
    tf.app.run()



















