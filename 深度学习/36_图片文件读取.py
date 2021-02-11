import os
import tensorflow as tf


def img_read(filelist):
    # 构建文件队列
    file_queue = tf.train.string_input_producer(filelist)
    # 定义reader
    reader = tf.WholeFileReader()
    k, v = reader.read(file_queue)  # 读取整个文件的内容
    # 解码
    img = tf.image.decode_jpeg(v)
    # 图片处理成统一的大小
    img_resized = tf.image.resize(img, [250, 250])  # 将图像设置成200*200
    """
    C<number_of_channels>----代表---一张图片的通道数,比如:

    channels = 1：灰度图片--grayImg---是--单通道图像

    channels = 3：RGB彩色图像---------是--3通道图像

    channels = 4：带Alph通道的RGB图像--是--4通道图像
    """
    img_resized.set_shape([250, 250, 3])  # 固定样本形状 ， 批处理时对形状有要求
    img_bat = tf.train.batch([img_resized],
                             batch_size=10,
                             num_threads=1)
    return img_bat


if __name__ == '__main__':
    dir_name = './test_img/'
    file_names = os.listdir(dir_name)
    file_list = []
    for f in file_names:
        file_list.append(os.path.join(dir_name, f))
    imgs = img_read(file_list)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 线程协调器
        threads = tf.train.start_queue_runners(sess, coord=coord)
        results = imgs.eval()  # 分批次读取样本
        # 等待线程结束，并回收寺院
        coord.request_stop()
        coord.join(threads)

# 显示图片
import matplotlib.pyplot as plt

plt.figure('Img Show', facecolor='lightgray')
for i in range(10):  # 循环显示读取到的样本
    plt.subplot(2, 5, i + 1)  # 显示2行5列的弟i+1个子图
    plt.xticks([])
    plt.yticks([])
    plt.imshow(results[i].astype('int32'))

plt.tight_layout()
plt.show()
