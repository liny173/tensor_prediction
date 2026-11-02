import pandas as pd

file_save = '数据上传间隔/interval=5/估计流量.csv'
link_lane = pd.read_csv('路段信息整合.csv')
link_lane_g = link_lane.groupby('Link ID')
e_data = pd.read_csv('数据上传间隔/interval=5/受控流量.csv')
s_dataframe = pd.read_csv('数据上传间隔/interval=5/抽样数量.csv')

# 获取所有唯一的 Link ID
link_ids = s_dataframe['Link ID'].unique()

# 遍历每个 Link ID 进行计算
new_rows = []
e_data['Link ID'] = e_data['Link ID'].astype(int)
e_data['Lane ID'] = e_data['Lane ID'].astype(int)

# 将 '周期流量' 列转换为四舍五入的整数类型
e_data['estimate_flow'] = e_data['estimate_flow'].round().astype(int)
s_dataframe['Link ID'] = s_dataframe['Link ID'].astype(int)
s_dataframe['Lane ID'] = s_dataframe['Lane ID'].astype(int)
s_dataframe['sample_flow'] = s_dataframe['sample_flow'].astype(int)

for link_id in link_ids:
    df1_link = s_dataframe[s_dataframe['Link ID'] == link_id]
    df2_link = e_data[e_data['Link ID'] == link_id]

    # 过滤相同 Link ID 和 Lane ID 的行
    df_merged = pd.merge(df1_link, df2_link, on=['Link ID', 'Lane ID'], suffixes=('_sample', '_estimate'))
    # print(df_merged)
    if not df_merged.empty:
        sam_stra = df_merged[df_merged['Lane ID'] == 2]['sample_flow'].iloc[0]
        est_stra = df_merged[df_merged['Lane ID'] == 2]['estimate_flow'].iloc[0]
        n1 = sam_stra / est_stra
        sam_left = df_merged[df_merged['Lane ID'] == 3]['sample_flow'].iloc[0]
        est_left = df_merged[df_merged['Lane ID'] == 3]['estimate_flow'].iloc[0]
        n2 = sam_left / est_left

        # 计算新的周期流量
        estimate_non = round(df1_link.loc[df1_link['Lane ID'] == 1, 'sample_flow'].values[0] / ((n1 + n2) / 2))

        # 创建新行
        new_row = {'Link ID': link_id, 'Lane ID': 1, 'estimate_flow': estimate_non}
        new_rows.append(new_row)

# 将新行添加到 dataframe2
dataframe2 = e_data._append(new_rows, ignore_index=True)
print(dataframe2)
dataframe2.to_csv(file_save,index=False)
