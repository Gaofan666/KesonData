import paddle.fluid as fluid

# 创建两个常量
x = fluid.layers.fill_constant(shape=[1], dtype='int64', value=5)
y = fluid.layers.fill_constant(shape=[1], dtype='int64', value=1)
z = x + y

# 创建执行器
place = fluid.CPUPlace()  # 指定程序再CPU上运行
exe = fluid.Executor(place)  # 创建执行器
result = exe.run(fluid.default_main_program(),
                 fetch_list=[z])  # 指定要返回的结果
print(result)
