import pandas as pd

chunk_size = 1000000  # 每次读取的行数
chunks = []
file_save = "数据上传间隔/interval=5/抽样数量.csv"
for chunk in pd.read_csv('抽样轨迹数据_间隔_5.csv', chunksize=chunk_size):
    chunks.append(chunk)
# 合并所有块
origin_data = pd.concat(chunks, ignore_index=True)
link_lane = pd.read_csv('路段信息整合.csv')

all_Q_list = []

# 受控流向
for index, row in link_lane.iterrows():
    Q = []
    link_id = row['Link ID']
    lane_id = row['Lane ID']
    Pos = row['Pos']
    stopline = row['Stopline']
    cycle_start = row['cycle_start']
    f_data = origin_data[
        (origin_data['Link ID'] == link_id) & (origin_data['Lane ID'] == lane_id)
        ]
    vehicle_count_per_hour = 0
    f_data = f_data[f_data['Timestep'] >= cycle_start]
    f_data['Timestep'] = f_data['Timestep'] - cycle_start
    s_data = f_data.groupby('Vehicle ID')
    for vehicle_id, group in s_data:
        # 对每个 group 按 'Timestep' 进行排序
        group = group.sort_values('Timestep')
        # 检查 'Front X' 或 'Front Y' 的值与 stopline 的差值绝对值是否小于等于1
        if Pos == 'X':
            if (abs(group['Front X'] - stopline) <= 3).any():
                vehicle_count_per_hour += 1
        elif Pos == 'Y':
            if (abs(group['Front Y'] - stopline) <= 3).any():
                vehicle_count_per_hour += 1
    Q.append({
        'Lane ID': lane_id,
        'Link ID': link_id,
        'sample_flow': vehicle_count_per_hour,
    })
    Q_df = pd.DataFrame(Q)
    all_Q_list.append(Q_df)

for index, row in link_lane.iterrows():
    Q = []
    link_id = row['Link ID']
    pos = row['Pos']
    stopline = row['Stopline']
    cycle_start = row['cycle_start']
    f_data = origin_data[
        (origin_data['Link ID'] == link_id) & (origin_data['Lane ID'] == 1)
        ]
    f_data = f_data[f_data['Timestep'] > cycle_start]
    f_data['Timestep'] = f_data['Timestep'] - cycle_start
    vehicle_count_per_hour = f_data['Vehicle ID'].nunique()
    Q.append({
        'Lane ID': 1,
        'Link ID': link_id,
        'sample_flow': vehicle_count_per_hour,
    })
    Q_df = pd.DataFrame(Q)
    all_Q_list.append(Q_df)

final_Q_list = pd.concat(all_Q_list, ignore_index=True)
final_Q_list = final_Q_list.drop_duplicates(subset=['Link ID', 'Lane ID'])
final_Q_list.to_csv(file_save, index=False)
