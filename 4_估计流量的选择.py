import pandas as pd


# 尝试不同的编码来读取CSV文件
def read_csv_with_encoding(file_path):
    encodings = ['utf-8', 'latin1', 'gbk']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取文件 {file_path}，请检查文件编码格式。")

file_save = '数据上传间隔/interval=1/受控流量.csv'
# 读取CSV文件
estimated_traffic = read_csv_with_encoding('数据上传间隔/interval=1/est_Q_FGD.csv')
real_traffic = read_csv_with_encoding('真实流量.csv')

# 创建一个空的DataFrame来存储结果
results = pd.DataFrame(columns=['Link ID', 'Lane ID', 'estimate_flow'])

# 遍历真实流量的每一行
for i, row in real_traffic.iterrows():
    link_id = row['Link ID']
    lane_id = row['Lane ID']
    real_flow = row['real_flow']

    # 找到估计流量中对应Link ID和Lane ID的行
    estimated_row = estimated_traffic[(estimated_traffic['Link ID'] == link_id) &
                                      (estimated_traffic['Lane ID'] == lane_id)]

    if not estimated_row.empty:
        # 提取该行的五个小时流量
        flows = estimated_row.iloc[0][
            ['est_flow_1', 'est_flow_2', 'est_flow_3', 'est_flow_4', 'est_flow_5']]
        # 找到与真实流量最接近的估计流量
        closest_flow = flows.iloc[(flows - real_flow).abs().argmin()]

        # 将结果存入新的DataFrame
        results = results._append({'Link ID': link_id, 'Lane ID': lane_id, 'estimate_flow': closest_flow},
                                  ignore_index=True)

# 保存结果为新的CSV文件
results.to_csv(file_save, index=False)
