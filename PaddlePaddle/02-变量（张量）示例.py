import paddle.fluid as fluid
import numpy

# 创建两个变量，2行3列，类型为浮点是
x = fluid.layers.data(name='x', shape=[2, 3], dtype='float32')
y = fluid.layers.data(name='y', shape=[2, 3], dtype='float32')

x_add_y = fluid.layers.elementwise_add(x, y)  # 张量按元素相加
x_mul_y = fluid.layers.elementwise_mul(x, y)  # 张量按元素相乘

place = fluid.CPUPlace()  # 指定再CPU上运行
exe = fluid.Executor(place)  # 执行器
exe.run(fluid.default_startup_program())  # 初始化

a = numpy.array([[1, 2, 3],
                 [4, 5, 6]])  # 2行3列的数组
b = numpy.array([[1, 1, 1],
                 [2, 2, 2]])
params = {'x': a, 'y': b}  # 参数字典

outs = exe.run(program=fluid.default_main_program(),  # 要执行的program
        feed=params,  # 执行program需要的参数
        fetch_list=[x_add_y, x_mul_y]  # 指定要获取的结果
        )

print(outs.shape)
for i in outs:
    print(i)