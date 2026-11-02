import pandas as pd

# 逐块读取 CSV 文件
chunk_size = 1000000  # 每次读取的行数
chunks = []
for chunk in pd.read_csv('抽样轨迹数据_0.1.csv', chunksize=chunk_size):
    chunks.append(chunk)
# 合并所有块
origin_data = pd.concat(chunks, ignore_index=True)
interval = 5  # 例如，每隔10秒保留一个数据点

# 按 'Vehicle ID' 分组
grouped = origin_data.groupby('Vehicle ID')
# 创建一个空的DataFrame用于存储结果
result = pd.DataFrame()

# 对每个 'Vehicle ID' 组进行操作
for vehicle_id, group in grouped:
    # 对每个组按照 'Timestep' 进行排序
    group = group.sort_values(by='Timestep')
    # 找到第一个非空的Timestep值
    first_timestep = group['Timestep'].dropna().iloc[0]
    # 计算开始索引，从第一个Timestep值开始，每隔interval秒取一个数据点
    start_index = int((first_timestep // interval) * interval)
    # 生成保留数据的Timestep索引列表
    keep_indices = range(start_index, int(group['Timestep'].max()) + 1, interval)
    # 使用Timestep索引列表筛选数据
    sampled_group = group[group['Timestep'].isin(keep_indices)]
    # 将筛选后的数据追加到结果DataFrame中
    result = pd.concat([result, sampled_group])
# 将过滤后的数据保存到新的 CSV 文件中
result.to_csv(f'抽样轨迹数据_间隔_{interval}.csv', index=False)
