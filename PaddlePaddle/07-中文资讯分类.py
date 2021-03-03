import os
from multiprocessing import cpu_count
import numpy as np
import paddle
import paddle.fluid as fluid

# 定义公共变量
data_root = './data/news_classify/'  # 数据集所在目录
data_file = 'news_classify_data.txt'  # 原始样本文件名称
test_file = 'test_file.txt'  # 测试集文件名称
train_file = 'train_list.txt'  # 训练集文件名称
dict_file = 'dict_txt.txt'  # 编码字典文件

data_file_path = data_root + data_file  # 样本文件完整路径
dict_file_path = data_root + dict_file  # 编码字典完整路径
test_file_path = data_root + test_file
train_file_path = data_root + train_file


# 生成字典文件:把每个字编码成唯一的数字，并存入文件
def creat_dict():
    dict_set = set()  # 集合，去重
    with open(data_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 遍历每行
    for line in lines:
        title = line.split('_!_')[-1].replace('\n', '')

        for w in title:  # 取出每个字
            dict_set.add(w)  # 将字加入集合去重

    # 遍历结合，为每个字分配一个编码
    dict_list = []
    i = 1  # 计数器
    for s in dict_set:
        dict_list.append([s, i])  # 将文字-编码 子列表添加到列表中
        i += 1
    dict_txt = dict(dict_list)  # 将列表直接转化为字典
    end_dict = {'<unk>': i}  # 未知字符
    dict_txt.update(end_dict)  # 将未知字符-编码键值对添加到字典中

    # 将字典对象保存到文件中
    with open(dict_file_path, 'w', encoding='utf-8') as f:
        f.write(str(dict_txt))  # 将字典对象转换成字符串并且存入文件
    print('生成字典完成')


# 对一行标题进行编码
def line_encoding(title, dict_txt, label):
    new_line = ''  # 编码后的结果
    for w in title:
        if w in dict_txt:  # 字在字典中，取出编码
            code = str(dict_txt[w])
        else:  # 未在字典中，取出未知字符编码
            code = str(dict_txt['<unk>'])
        new_line = new_line + code + ','
    new_line = new_line[:-1]  # 去掉最后一个逗号
    new_line = new_line + '\t' + label + '\n'  # 拼接编码后的字符串及类别
    return new_line


# 对原始样本进行编码，将编码后的字符串存入测试集和训练集
def create_data_list():
    # 清空测试集 训练集
    with open(test_file_path, 'w') as f:
        pass
    with open(train_file_path, 'w') as f:
        pass

    # 打开原始样本，取出每个样本的标题进行编码
    with open(data_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(dict_file_path, 'r', encoding='utf-8') as f:
        # 读取字典文件的第一行  实际上只有一行 并当作表达式执行，返回一个字典对象
        dict_txt = eval(f.readlines()[0])
    # 遍历样本的每一行，取出标题部分编码
    i = 0
    for line in lines:
        words = line.replace('\n', '').split('_!_')  # 拆分每行
        label = words[1]  # 类别
        title = words[3]  # 标题

        new_line = line_encoding(title, dict_txt, label)

        if i % 10 == 0:  # 写测试集
            with open(test_file_path, 'a', encoding='utf-8') as f:
                f.write(new_line)
        else:  # 写训练集
            with open(train_file_path, 'a', encoding='utf-8') as f:
                f.write(new_line)
        i += 1
    print('生成测试集 训练集结束')


creat_dict()  # 生成字典
create_data_list()  # 编码 生成测试集 训练集


##############################  模型搭建 ，训练  评估   #############################################################

def get_dict_len(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        line = eval(f.readlines()[0])
    return len(line.keys())  # 返回字典对象的长度


# data_mapper: 将传入的一行样本转换为整型列表并返回
def data_mapper(sample):
    data, label = sample  # 将sample元组拆分到两个变量中
    # 拆分每个编码后的值 并转换为整数，生成一个列表
    val = [int(w) for w in data.split(',')]
    return val, int(label)  # 返回整型列表和标签(转换为整数)


# reader
def train_reader(train_file_path):
    def reader():
        with open(train_file_path, 'r') as f:
            lines = f.readlines()
            np.random.shuffle(lines)  # 打乱样本顺序，做随机化处理

            for line in lines:
                data, label = line.split('\t')
                yield data, label

    return paddle.reader.xmap_readers(data_mapper,  # reader读取的数据进行下一步处理的函数
                                      reader,  # 读取样本函数
                                      cpu_count(),  # 线程数量
                                      1024)  # 缓冲区的大小


def test_reader(test_file_path):
    def reader():
        with open(test_file_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                data, label = line.split('\t')
                yield data, label

    return paddle.reader.xmap_readers(data_mapper,
                                      reader,
                                      cpu_count(),
                                      1024)


# 定义网络结构
def CNN_net(data, dict_dim, class_dim=10, emb_dim=128, hid_dim=128, hid_dim2=98):
    """
    搭建TextCNN模型
    :param data:  原始数据
    :param dict_dim: 词典大小
    :param class_dim: 分类数量
    :param emb_dim: 词嵌入计算参数
    :param hid_dim: 第一组卷积运算卷积核数量
    :param hid_dim2: 第二组卷积运算卷积核数量
    :return: 运算的结果
    """
    # embedding(词嵌入层)：生成词向量，得到一个粘稠的实向量表示
    # 能以最少的维度，表达丰富的文本信息
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])

    # 并列两组卷积，池化
    conv1 = fluid.nets.sequence_conv_pool(input=emb,  # 输入，词嵌入层的输出
                                          num_filters=hid_dim,  # 卷积核数量
                                          filter_size=3,  # 卷积核大小(在前后三个字或词之间提取局部特征)
                                          act='tanh',  # 激活函数
                                          pool_type='sqrt')  # 池化类型
    conv2 = fluid.nets.sequence_conv_pool(input=emb,  # 输入，词嵌入层的输出
                                          num_filters=hid_dim2,  # 卷积核数量
                                          filter_size=4,  # 卷积核大小（图像处理一般不选择双数）
                                          act='tanh',  # 激活函数
                                          pool_type='sqrt')  # 池化类型
    # 输出层
    output = fluid.layers.fc(input=[conv1, conv2],  # 前面两组卷积层的输出作为输入
                             size=class_dim,  # 分类数量
                             act='softmax')  # 激活函数
    return output


# 定义变量
model_save_dir = 'model/news_classify/'  # 模板保存路径

words = fluid.layers.data(name='words', shape=[1], dtype='int64',
                          lod_level=1)  # 张量层级
label = fluid.layers.data(name='label', shape=[1], dtype='int64')  # 标签
# 获取字典长度
dict_dim = get_dict_len(dict_file_path)
# 调用函数创建网络
model = CNN_net(words, dict_dim)
# 损失函数
cost = fluid.layers.cross_entropy(input=model,  # 预测值
                                  label=label)  # 真实值
avg_cost = fluid.layers.mean(cost)
# 准确率
acc = fluid.layers.accuracy(input=model,  # 预测结果
                            label=label)  # 真实结果

# 克隆一个program用于模型评估
test_program = fluid.default_main_program().clone(for_test=True)
# 优化器
optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.001)
optimizer.minimize(avg_cost)
# 执行器
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
# 调用reader函数创建读取器
tr_reader = train_reader(train_file_path)  # 训练集原始读取器
batch_train_reader = paddle.batch(tr_reader, batch_size=128)  # 批量读取器

ts_reader = test_reader(test_file_path)  # 测试集原始读取器
batch_test_reader = paddle.batch(ts_reader, batch_size=128)  # 批量读取器
# feeder
feeder = fluid.DataFeeder(place=place,
                          feed_list=[words, label])  # 需要喂入的参数

# 开始训练
for pass_id in range(5):
    for batch_id, data in enumerate(batch_train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        # 打印
        if batch_id % 100 == 0:
            print('pass_id:%d,batch_id:%d,cost:%f,acc:%f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))
    # 模型评估
    test_costs_list = []
    test_accs_list = []

    for batch_id, data in enumerate(batch_test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, acc])
        test_costs_list.append(test_cost[0])  # 记录损失值
        test_accs_list.append(test_acc[0])
    # 计算测试集下平均损失值，准确率
    avg_test_cost = sum(test_costs_list) / len(test_costs_list)
    avg_test_acc = sum(test_accs_list) / len(test_accs_list)

    print('pass_id:%d,test_cost:%f,test_acc:%f' %
          (pass_id, avg_test_cost, avg_test_acc))

# 训练结束 保存模型
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
fluid.io.save_inference_model(model_save_dir,
                              feeded_var_names=[words.name],
                              target_vars=[model],
                              executor=exe)
print('模型保存完成')

# -----------------------------------模型加载和预测------------------------

model_save_dir = 'model/news_classify/'


def get_data(sentence):  # 对待预测文本编码
    # 读取字典内容
    with open(dict_file_path, 'r', encoding='utf-8') as f:
        dict_txt = eval(f.readlines()[0])
    keys = dict_txt.keys()
    ret = []  # 编码后的结果
    for s in sentence:
        if not s in keys:
            s = '<unk>'
        ret.append(int(dict_txt[s]))
    return ret  # 返回经过编码后的列表


# 执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 加载模型
print('加载模型')
infer_program, feeded_var_names, target_var = \
    fluid.io.load_inference_model(dirname=model_save_dir,
                                  executor=exe)

# 生成一批测试数据
texts = []
data1 = get_data("在获得诺贝尔文学奖7年之后，莫言15日晚间在山西汾阳贾家庄如是说")
data2 = get_data("综合'今日美国'、《世界日报》等当地媒体报道，芝加哥河滨警察局表示")
data3 = get_data("中国队无缘2020年世界杯")
data4 = get_data("中国人民银行今日发布通知，降低准备金率，预计释放4000亿流动性")
data5 = get_data("10月20日,第六届世界互联网大会正式开幕")
data6 = get_data("同一户型，为什么高层比低层要贵那么多？")
data7 = get_data("揭秘A股周涨5%资金动向：追捧2类股，抛售600亿香饽饽")
data8 = get_data("宋慧乔陷入感染危机，前夫宋仲基不戴口罩露面，身处国外神态轻松")
data9 = get_data("此盆栽花很好养，花美似牡丹，三季开花，南北都能养，很值得栽培")  # 不属于任何一个类别

texts.append(data1)
texts.append(data2)
texts.append(data3)
texts.append(data4)
texts.append(data5)
texts.append(data6)
texts.append(data7)
texts.append(data8)
texts.append(data9)

# 获取每个句子词数量
base_shape = [[len(c) for c in texts]]
# 生成LodTensor
tensor_words = fluid.create_lod_tensor(texts, base_shape, place)
# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: tensor_words},  # 待预测的数据
                 fetch_list=target_var)

names = ["文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "国际", "证券"]

# 获取最大值的索引
for i in range(len(texts)):
    lab = np.argsort(result)[0][i][-1]  # 取出最大值的元素下标  默认升序排序
    print("预测结果：%d, 名称:%s, 概率:%f" % (lab, names[lab], result[0][i][lab]))
