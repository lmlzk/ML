import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

#
# def main(argv):
#
#     # 加载数据
#     mnist = input_data.read_data_sets("./tmp/mnist/input_data/", one_hot=True)
#
#     # 1、准备数据，特征值matrix x[None, 784] ，标签值  y  [None, 10]
#     with tf.variable_scope("data"):
#
#         # 准备特征值占位符
#         x = tf.placeholder(tf.float32, [None, 784], name="x_data")
#
#         # 目标值占位符
#         y_true = tf.placeholder(tf.int32, [None, 10], name="y_true")
#
#     # 2、随机指定权重，偏置，w[784, 10]   b[10],得出加权之后的结果
#     with tf.variable_scope("model"):
#
#         # 准备权重变量 matrix->[784,10]
#         W = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name="weight")
#
#         # 准备偏置
#         b = tf.Variable(tf.constant(0.0, shape=[10]), name="bias")
#
#         # 计算模型结果, [None,784] * [784, 10] + [10]
#         y_predict = tf.matmul(x, W) + b
#
#     # 3、softmax计算，损失值计算，计算平均损失
#     with tf.variable_scope("compute_loss"):
#
#         # 取平均损失
#         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
#
    # # 4、梯度下降优化损失
    # with tf.variable_scope("SGD"):
    #
    #     # 优化
    #     train_op = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
    #
    # # 5、计算准确率
    # equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
    #
    # accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
#
#     # 收集变量
#     tf.summary.scalar("loss", loss)
#
#     tf.summary.scalar("accuracy", accuracy)
#
#     tf.summary.histogram("weight", W)
#
#     tf.summary.histogram("biases", b)
#
#     # 合并变量
#     merged = tf.summary.merge_all()
#
#     # 初始化变量op
#     init_op = tf.global_variables_initializer()
#
#     with tf.Session() as sess:
#         # 运行初始化
#         sess.run(init_op)
#
#         # 建立事件文件
#         file_writer = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)
#
#         # 循环优化损失
#         for i in range(2000):
#
#             # 获取数据 mnist_x, mnist_y
#             mnist_x, mnist_y = mnist.train.next_batch(50)
#
#             # 运行优化器
#             sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
#
#             summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})
#
#             # 运行合并结果
#             file_writer.add_summary(summary, i)
#
#             # 打印准确率
#             print(sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y}))


# 生成随机权重
def weight_variable(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w


# 生成随机偏置
def bias_variable(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b


def model():

    # 准备数据占位符 x [None, 784] y [None, 10]
    with tf.variable_scope("data"):

        x_data = tf.placeholder(tf.float32, [None, 784])

        y_true = tf.placeholder(tf.int32, [None, 10])

    # 卷积层1, 32个filter,5*5,步长1，padding"SAME"
    with tf.variable_scope("conv1"):

        # 准备权重和偏置,w ->[5, 5, 1, 32], b ->[32]
        w_conv1 = weight_variable([5, 5, 1, 32])

        b_conv1 = bias_variable([32])

        # 处理输入图片数据的形状，卷积层要求
        x_reshape = tf.reshape(x_data, [-1, 28, 28, 1])

        # 进行卷积，激活，池化[None, 28, 28, 32]-->池化  卷积层1的输出：[None, 14, 14, 32]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)

        # 池化（减少特征）->[None, 14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 卷积层2 64个filter,5*5,步长1，padding"SAME"
    with tf.variable_scope("conv2"):

        # 准备权重和偏置,w [5, 5, 32, 64], b->[64]
        w_conv2 = weight_variable([5, 5, 32, 64])

        b_conv2 = bias_variable([64])

        # 进行卷积、激活、池化
        # 输入[None, 14, 14, 32] 卷积：[None, 14, 14, 64] 经过池化层输出：[None, 7, 7, 64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)

        # x_pool2 ->[None, 7, 7, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 全连接层[None, 7*7*64] * [7*7*64, 10] + [10] = [None, 10]<--模型输出结果，预测值
    with tf.variable_scope("FC"):

        # 准备权重和偏置
        w_fc = weight_variable([7 * 7 * 64, 10])

        b_fc = bias_variable([10])

        # 矩阵变换[None, 7, 7, 64]-->[None, 7*7*64]
        x_fc_reshape = tf.reshape(x_pool2, [-1, 7 * 7 * 64])

        # 进行全连接层计算，得出结果
        y_predict = tf.matmul(x_fc_reshape, w_fc) + b_fc

    return x_data, y_true, y_predict


def compute(y_true, y_predict):
    """
    计算sofmax,交叉熵损失
    :param y_true: 真实值
    :param y_predict: 预测值
    :return: 损失
    """
    #3、softmax计算，损失值计算，计算平均损失
    with tf.variable_scope("compute_loss"):

        # 取平均损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    return loss


def sgd(loss, y_true, y_predict):
    """
    优化损失，计算准确率
    :param loss: 损失
    :param y_true: 真实值
    :param y_predict: 预测值
    :return: train_op, accuracy
    """
    with tf.variable_scope("SGD"):
        # 优化
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # 5、计算准确率
    equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))

    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    return  train_op, accuracy



def main(argv):

    mnist = input_data.read_data_sets("./tmp/mnist/input_data/", one_hot=True)

    # 建立卷积神经网络模型,返回预测结果，真实数据的占位符
    x_data, y_true, y_predict = model()

    # 进行softmax, 交叉熵损失计算
    loss = compute(y_true, y_predict)

    # 梯度下降优化，计算准确率
    train_op, accuracy = sgd(loss, y_true, y_predict)

    # 定义变量op
    init_op = tf.global_variables_initializer()

    # 通过会话循环训练
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 循环训练
        for i in range(2000):

            # 获取特征值，目标值数据
            mnist_x, mnist_y = mnist.train.next_batch(50)

            # 打印准确率
            if i % 100 == 0:
                print("训练的准确率：", sess.run(accuracy, feed_dict={x_data: mnist_x, y_true:mnist_y}))

            # 运行优化器
            sess.run(train_op, feed_dict={x_data: mnist_x, y_true:mnist_y})


if __name__ == "__main__":
    tf.app.run()
















