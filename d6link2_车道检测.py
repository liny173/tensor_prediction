import hdbscan
import jenkspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 载入某一进口道范围内的数据
# DataFrame结构：每行一个车辆唯一ID，不同时间点对应不同经纬度
df = pd.read_csv(r"D:\01\traffic\tensor prediction\data_4\预处理数据_20181029_d6_1000_1030_22.csv")

# Step 1: 预处理数据 - 提取每辆车的最后位置数据
# # 将时间转换为datetime格式，以便后续排序
# 按每个VehicleID取时间最大的记录
df_latest = df.sort_values('Time').groupby('ID').tail(1).reset_index(drop=True)

# Step 2: 计算到道路边缘的距离（假设道路边缘的经纬度为固定点）
road_edge_lat, road_edge_lon = 37.985840, 23.729753  # 道路边缘点


def calculate_distance(lat1, lon1, lat2, lon2):
    # 使用haversine公式计算两点之间的距离
    R = 6371000  # 地球半径，单位为米
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


# 计算每辆车到道路边缘的距离
df_latest['distance_from_edge'] = df_latest.apply(
    lambda row: calculate_distance(row['Latitude'], row['Longitude'], road_edge_lat, road_edge_lon), axis=1
)

# 使用HDBSCAN算法进行初步车道分组，目的是为了去掉噪声点
# 调整HDBSCAN的参数以适应数据的分布
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5)
df_latest['Lane'] = clusterer.fit_predict(df_latest[['Latitude', 'Longitude']])

# 过滤掉噪声点（-1表示噪声点）
df_latest = df_latest[df_latest['Lane'] != -1]

# Step 3: 使用詹克斯优化算法进行车道分组
# 为每个车辆分配车道
def assign_lane(distance):
    for i in range(1, len(breaks)):
        if breaks[i - 1] <= distance < breaks[i]:
            return i
    return len(breaks) - 1  # 如果在最后一个分段


num_lanes = 2  # 根据谷歌地图得出划分成4个车道
breaks = jenkspy.jenks_breaks(df_latest['distance_from_edge'].values, num_lanes)
df_latest['Lane'] = df_latest['distance_from_edge'].apply(assign_lane)


# Step 4: 检查每个车道的宽度是否符合要求
def check_lane_widths(df, min_width=2.5, max_width=4.0):
    for lane in df['Lane'].unique():
        lane_data = df[df['Lane'] == lane]
        max_distance = lane_data['distance_from_edge'].max()
        min_distance = lane_data['distance_from_edge'].min()
        lane_width = max_distance - min_distance
        if not (min_width <= lane_width <= max_width):
            return False
    return True


# Step 5: 若车道宽度不符合要求，增加车道数重新分配
max_lanes = 4  # 假设的最大车道数
while not check_lane_widths(df_latest) and num_lanes < max_lanes:
    num_lanes += 1
    breaks = jenkspy.jenks_breaks(df_latest['distance_from_edge'].values, num_lanes)
    df_latest['Lane'] = df_latest['distance_from_edge'].apply(assign_lane)

print(f"Detected number of lanes: {num_lanes}")
print(df_latest[['ID', 'Latitude', 'Longitude', 'Lane']])

# 设置绘图风格
sns.set(style="whitegrid")

# 绘制边缘距离的频率直方图
plt.figure(figsize=(12, 6))
sns.histplot(df_latest['distance_from_edge'], bins=30, kde=True, color='skyblue')
plt.title('Histogram of Distance from Road Edge')
plt.xlabel('Distance from Edge (meters)')
plt.ylabel('Frequency')
plt.show()

# 绘制车道编号的频率直方图
plt.figure(figsize=(12, 6))
sns.histplot(df_latest['Lane'], bins=len(df_latest['Lane'].unique()), discrete=True, color='salmon')
plt.title('Histogram of Lane Distribution')
plt.xlabel('Lane')
plt.ylabel('Frequency')
plt.show()

df_merge = pd.merge(df, df_latest[['ID', 'Lane']], on='ID', how='inner')
df_merge['Link'] = 22
df_merge.to_csv('D:\\01\\traffic\\tensor prediction\data_4\车道Lane数据_22.csv', index=False)

