# 原始读取器
import paddle

def reader_creator(file_path):
    def reader():
        with open(file_path, 'r') as f:
            lines = f.readlines()  # 读取所有行
            for line in lines:
                yield line  # 利用生成器关键字创建一个数据并返回

    return reader


reader = reader_creator('test.txt')
shuffle_reader = paddle.reader.shuffle(reader,10) # 随机化处理
batch_reader = paddle.batch(shuffle_reader,3) # 批量随机读取器  3个3个读取
# for data in reader():
#     print(data, end='')

# for data in shuffle_reader():
#     print(data,end="")

for data in batch_reader():
    print(data,end="")