# 读取CSV文件

import tensorflow as tf
import os



def csv_read(filelist):
    # 构建文件队列
    file_queue = tf.train.string_input_producer(filelist)
    # 定义reader
    reader = tf.TextLineReader()
    k, v = reader.read(file_queue)  # 读取，返回文件名称，数据
    # 解码
    records = [['None'], ['None']]
    example, label = tf.decode_csv(v, record_defaults=records)
    # 批处理  返回二进制的数据和标签 批次大小决定了返回数据的多少
    example_bat, label_bat = tf.train.batch([example, label],  # 参与批处理的数据
                                            batch_size=9,  # 批次大小
                                            num_threads=1)  # 线程数量
    return example_bat, label_bat


if __name__ == '__main__':
    # 构建文件列表
    dir_name = './test_data/'
    file_names = os.listdir(dir_name)  # 列出目录下所有的文件
    file_list = []
    for f in file_names:
        # 将目录名称，文件名称拼接成完整路径，并添加到文件列表
        file_list.append(os.path.join(dir_name, f))
    # 调用自定义函数，读取指定文件列表中的数据
    example, label = csv_read(file_list)

    # 开启Session,执行
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 定义线程协调器
        threads = tf.train.start_queue_runners(sess, coord=coord)
        print(sess.run([example, label]))  # 执行操作

        # 等待线程停止，并回收资源
        coord.request_stop()
        coord.join(threads)
