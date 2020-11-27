import numpy as np

# 中国联通用户流失情况分析  这种分析方法为 分组 分析

# 加载数据
with open('CustomerSurvival.csv', 'r') as f:
    data = []
    # index 行数索引(0，1，2...)  line 每一行的数据（包括了i）[i,数据]
    for index, line in enumerate(f.readlines()):
        # print(line)
        # print(line[:-1])
        # print(line[:-1].split(','))
        # 将每一行的数据按照逗号拆分并且存到元组当中，放到Data当中
        row = tuple(line[:-1].split(','))  # line[:-1]是数据
        data.append(row)

    # 转成ndarry
    # pack_type 套餐金额（1.<96 2.96-225 3.>225）
    # pack_change 改变行为(是否曾经改变过套餐金额 1.是 0.否)
    # contract 服务合约：用户是否与联通签订过服务合约 1.是 0.否
    # asso_pur 关联购买：用户再使用联通移动服务过程中是否同时办理其他业务
    # （主要是固定电话和宽带业务 1.同时办理一项其他业务 2.同时办理两项其他业务）
    # group_user 集团用户(联通内部人员)
    data = np.array(data, dtype={
        'names': ['index', 'pack_type', 'extra_time', 'extra_flow', 'pack_change',
                  'contract', 'asso_pur', 'group_user', 'use_month', 'loss'],
        'formats': ['i4', 'i4', 'f8', 'f8', 'i4', 'i4', 'i4', 'i4', 'i4', 'i4']
    })
    print(data)

# 统计流失用户与未流失用户的占比
loss_data = data[data['loss'] == 1]  # 流失的用户
unloss_data = data[data['loss'] == 0]  # 未流失的用户
print('流失用户比例：', len(loss_data) / len(data))
print('未流失用户比例：', len(unloss_data) / len(data))

# 分析 额外通话时长
print('流失用户剩余通话时长(平均)', loss_data['extra_time'].mean())
print('未流失用户剩余通话时长(平均)', unloss_data['extra_time'].mean())

# 分析 额外流量
print('流失用户额外流量(平均)', loss_data['extra_flow'].mean())
print('未流失用户额外流量(平均)', unloss_data['extra_flow'].mean())

# 分析套餐类型 pack_type
types = set(data['pack_type'])  # 离散值 利用set集合拿到不重复的值
print('-' * 50)
print('套餐类型：', types)
for type in types:
    # 获取每种类型的数据量，看一下占比
    sub_data = data[data['pack_type'] == type]  # 选择该套餐的人的数据集
    print('type:', type, '占比：', len(sub_data) / len(loss_data))
    loss_data = sub_data[sub_data['loss'] == 1]  # 该套餐的人数中的流失用户
    unloss_data = sub_data[sub_data['loss'] == 0]  # 该套餐的人数中的未流失用户
    print('type:', type, '流失用户占比:', len(loss_data) / len(sub_data))
    print('type:', type, '未流失用户占比:', len(unloss_data) / len(sub_data))

# 分析套餐改变行为对流失率的影响
types = set(data['pack_change'])
print('-' * 50)
print('改变行为：', types)
for type in types:
    sub_data = data[data['pack_change'] == type]
    print('type:', type, '占比:', len(sub_data) / len(data))
    loss_data = sub_data[sub_data['loss'] == 1]
    unloss_data = sub_data[sub_data['loss'] == 0]
    print('type:', type, '流失用户占比：', len(loss_data) / len(sub_data))
    print('type', type, '未流失用户占比：', len(unloss_data) / len(sub_data))

# 分析 关联购买 asso_pur 对流失率的影响
types = set(data['asso_pur'])
print('-' * 50)
print('关联购买：', types)
for type in types:
    sub_data = data[data['asso_pur'] == type]
    print('type:', type, '占比:', len(sub_data) / len(data))
    loss_data = sub_data[sub_data['loss'] == 1]
    unloss_data = sub_data[sub_data['loss'] == 0]
    print('type:', type, '流失用户占比：', len(loss_data) / len(sub_data))
    print('type', type, '未流失用户占比：', len(unloss_data) / len(sub_data))

# 分析 集团用户group_user 对流失率的影响
types = set(data['group_user'])
print('-' * 50)
print('集团用户：', types)
for type in types:
    sub_data = data[data['group_user'] == type]
    print('type:', type, '占比:', len(sub_data) / len(data))
    loss_data = sub_data[sub_data['loss'] == 1]
    unloss_data = sub_data[sub_data['loss'] == 0]
    print('type:', type, '流失用户占比：', len(loss_data) / len(sub_data))
    print('type', type, '未流失用户占比：', len(unloss_data) / len(sub_data))
