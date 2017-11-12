import tensorflow as tf
import os


# 完成去出队列内容，+ 1， 出队列任务


# # 1、创建队列
# Q = tf.FIFOQueue(3, tf.float32)
#
# # 2、往队列里面填充一些数据
# init = Q.enqueue_many([[0.1, 0.2, 0.3], ])
#
# # 3、取出数据，+1，放回队列
# out_q = Q.dequeue()
#
# # 这个加号，由于有一个元素是op，所以被重载
# data = out_q + 1
#
# en_q = Q.enqueue(data)
#
# # 会话运行队列操作
# with tf.Session() as sess:
#     # 初始化队列
#     sess.run(init)
#
#     # 循环几次第三个步骤里面的操作
#     for i in range(2):
#         sess.run(en_q)
#
#     for i in range(Q.size().eval()):
#         print(sess.run(Q.dequeue()))


# 实现异步读取队列内容

# # 1、定义队列大小1000，队列里的数据类型
# Q = tf.FIFOQueue(1000, tf.float32)
#
# # 2、定义变量0.0， 自增之后的数据放入队列当中
# var = tf.Variable(0.0, tf.float32)
#
# data = tf.assign_add(var, tf.constant(1.0))
#
# en_q = Q.enqueue(data)
#
# # 3、线程管理器定义线程个数，线程执行的队列操作
# qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 3)
#
# init = tf.global_variables_initializer()
#
# # 会话掌握了队列资源，主线程运行结束，资源没有了，子线程运行就会报错
# with tf.Session() as sess:
#     sess.run(init)
#
#     # 定义一个线程协调器
#     coord = tf.train.Coordinator()
#
#     # 创建线程，往队列当中填充数据
#     threads = qr.create_threads(sess, coord=coord, start=True)
#
#     # 主线程逻辑去出队列的数据
#     for i in range(200):
#         print(sess.run(Q.dequeue()))
#
#     # 请求子线程停止，回收资源
#     coord.should_stop()
#
#     coord.join(threads)


# csv文件读取


# def csvreader(file_list):
#     """
#     读取CSV文件程序
#     :return: 样本的特征值，目标值
#     """
#     # 1、构造文件队列, 返回文件队列
#     file_queue = tf.train.string_input_producer(file_list)
#
#     # 2、生成文件读取器，读取队列的文件当中的内容
#     reader = tf.TextLineReader()
#
#     # key是文件名，value文件一行内容
#     key, value = reader.read(file_queue)
#
#     # 3、进行解码,指定默认值和类型，嵌套列表[[1],[1.0],[2.0]]
#     records = [["NULL"], ["NULL"]]
#
#     # 按照列数内容返回
#     example, label = tf.decode_csv(value, record_defaults=records)
#
#     # 进行批处理
#     example_batch, label_batch = tf.train.batch([example, label], batch_size=9, num_threads=1, capacity=20)
#
#     return example_batch, label_batch
#
#
# if __name__ == "__main__":
#     # 找到文件路径，名字，构造文件队列 "A.csv"...
#     filename = os.listdir("./data/csvdata/")
#
#     # 加上路径
#     file_list = [os.path.join("./data/csvdata/", file) for file in filename]
#
#     example, label = csvreader(file_list)
#
#     # 会话当中运行结果，线程同步，开启线程
#     with tf.Session() as sess:
#         # 定义一个线程协调器
#         coord = tf.train.Coordinator()
#
#         # 开启线程去运行队列的操作（往队列假数据，读取，解码）
#         threads = tf.train.start_queue_runners(sess, coord=coord)
#
#         # 打印读取的值
#         print(sess.run([example, label]))
#
#         # 使用强制停止运行
#         coord.request_stop()
#
#         coord.join(threads)


# 读取图片文件
#
# def picread(file_list):
#     """
#     读取狗图片并转换成 张量
#     :return:
#     """
#     # 1、构造文件的队列
#     file_queue = tf.train.string_input_producer(file_list)
#
#     # 2、生成图片读取器，读取队列内容
#     reader = tf.WholeFileReader()
#
#     key, value = reader.read(file_queue)
#
#     print(value)
#
#     # 3、进行图片的解码
#     image = tf.image.decode_jpeg(value)
#
#     print(image)
#
#     # 4、处理图片大小
#     image_resize = tf.image.resize_images(image, [256, 256])
#
#     print(image_resize)
#
#     # 设置静态形状
#     image_resize.set_shape([256, 256, 3])
#
#     print(image_resize)
#
#     # 5、进行批处理
#     image_batch = tf.train.batch([image_resize], batch_size=100, num_threads=1, capacity=100)
#
#     print(image_batch)
#     return image_batch
#
#
# if __name__ == "__main__":
#
#     # 找到文件路径，名字，构造文件队列 "A.csv"...
#
#     filename = os.listdir("./data/dog/")
#
#     # 加上路径
#     file_list = [os.path.join("./data/dog/", file) for file in filename]
#
#     image_batch = picread(file_list)
#
#     # 会话
#     with tf.Session() as sess:
#         # 定义线程协调器
#         coord = tf.train.Coordinator()
#
#         # 开启线程
#         threads = tf.train.start_queue_runners(sess, coord=coord)
#
#         print(sess.run(image_batch))
#
#         # 回收线程
#         coord.request_stop()
#
#         coord.join(threads)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("tfrecord_dir", "./tmp/cifar10.tfrecords", "写进图片数据文件的文件名")


# 读取二进制转换文件
class CifarRead(object):
    """
    读取二进制文件转换成张量，写进TFRecords，同时读取TFRecords
    """
    def __init__(self, file_list):
        """
        初始化图片参数
        :param file_list: 图片的路径名称列表
        """
        # 文件列表
        self.file_list = file_list

        # 图片大小，二进制文件字节数
        self.height = 32
        self.width = 32
        self.channel = 3
        self.label_bytes = 1
        self.image_bytes = self.height * self.width * self.channel
        self.bytes = self.label_bytes + self.image_bytes

    def read_and_decode(self):
        """
        解析二进制图片到张量
        :return: 批处理的image,label张量
        """
        # 1、构造文件队列
        file_queue = tf.train.string_input_producer(self.file_list)

        # 2、阅读器读取内容
        reader = tf.FixedLengthRecordReader(self.bytes)

        key, value = reader.read(file_queue)

        print(value)

        # 3、进行解码，处理格式
        label_image = tf.decode_raw(value, tf.uint8)

        print(label_image)

        # 处理格式，image,label
        # 进行切片处理,标签值
        label = tf.cast(tf.slice(label_image, [0], [self.label_bytes]), tf.int32)

        # 处理图片数据
        image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])

        print(image)

        # 处理图片的形状（提供给批处理）
        image_tensor = tf.reshape(image, [self.height, self.width, self.channel])

        print(image_tensor)

        # 批处理图片数据
        image_batch, label_batch = tf.train.batch([image_tensor, label], batch_size=10, num_threads=1, capacity=10)

        return image_batch, label_batch

    def write_to_tfrecords(self, image_batch, label_batch):

        # 建立TFRecords文件存储器
        writer = tf.python_io.TFRecordWriter(FLAGS.tfrecord_dir)

        # 循环取出每个样本的值，构造example
        for i in range(10):

            # 取出图片值,写进去的是值不是tensor类型
            image = image_batch[i].eval().tostring()

            # 取出标签值
            label = int(label_batch[i].eval()[0])

            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))

            # 写进序列化后的值
            writer.write(example.SerializeToString())

        writer.close()
        return None

    def read_from_tfrecords(self):
        """
        从TFRecords文件当中读取图片数据(解析example)
        :return: image_batch, label_batch
        """
        # 1、构造文件队列
        file_queue = tf.train.string_input_producer([FLAGS.tfrecord_dir])

        # 构造阅读器
        reader = tf.TFRecordReader()

        key, value = reader.read(file_queue)

        # 解析协议块
        feature = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })

        # feature["image"]    feature["label"]
        label = tf.cast(feature["label"], tf.int32)

        # 处理图片数据,由于是一个string，要进行解码
        image = tf.decode_raw(feature["image"], tf.uint8)

        image_tensor = tf.reshape(image, [self.height, self.width, self.channel])

        # 批处理

        image_batch, label_batch = tf.train.batch([image_tensor, label], batch_size=10, num_threads=1, capacity=10)

        return image_batch, label_batch


if __name__ == "__main__":
    # 找到文件路径，名字，构造文件队列 "A.csv"...

    filename = os.listdir("./data/cifar10/cifar-10-batches-bin/")

    # 加上路径
    file_list = [os.path.join("./data/cifar10/cifar-10-batches-bin/", file) for file in filename if file[-3:] == "bin"]

    # 初始化参数
    cr = CifarRead(file_list)

    # image_batch, label_batch = cr.read_and_decode()

    image_batch, label_batch = cr.read_from_tfrecords()

    with tf.Session() as sess:
        # 线程协调器
        coord = tf.train.Coordinator()

        # 开启线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        print(sess.run([image_batch, label_batch]))

        # print("存进TFRecords文件中")

        # cr.write_to_tfrecords(image_batch, label_batch)

        # print("存进文件完毕")


        coord.request_stop()

        coord.join(threads)



































