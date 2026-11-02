import random
import pandas as pd

# 逐块读取 CSV 文件
chunk_size = 1000000  # 每次读取的行数
chunks = []
for chunk in pd.read_csv('原始轨迹数据.csv', chunksize=chunk_size):
    chunks.append(chunk)
# 合并所有块
origin_data = pd.concat(chunks, ignore_index=True)

unique_vehicle_ids = origin_data['Vehicle ID'].unique()
# 随机选择要保留的 Vehicle_ID

# 采用总体简单抽样法抽取
# 修改渗透率p
p = 0.05
sampled_vehicle_ids = random.sample(list(unique_vehicle_ids), k=int(p * len(unique_vehicle_ids)))

# 过滤数据，只保留选定的 Vehicle_ID 内容的记录
sampled_data = origin_data[origin_data['Vehicle ID'].isin(sampled_vehicle_ids)]
# 将过滤后的数据保存到新的 CSV 文件中
sampled_data.to_csv(f'抽样轨迹数据_{p}.csv', index=False)
