import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
chunk_size = 1000000  # 每次读取的行数
chunks = []
for chunk in pd.read_csv(r"..\抽样轨迹数据_0.5.csv", chunksize=chunk_size):
    chunks.append(chunk)
# 合并所有块
sampled_data = pd.concat(chunks, ignore_index=True)

# 创建行名列表
rows = [f"{i}-{j}" for i in range(1, 9) for j in range(1, 9)]
data1 = {'path1': [0] * 64, 'path2': [0] * 64}
data2 = {'pathlink1': [0] * 64, 'pathlink2': [0] * 64}
df1 = pd.DataFrame(data1, index=rows)
df2 = pd.DataFrame(data2, index=rows)
file_path1 = r"流向路径表格.xlsx"
file_path2 = r"路段路径表格.xlsx"
df_matrix1 = pd.read_excel(file_path1)
df_matrix1 = df_matrix1.drop('Unnamed: 0', axis=1)
df_matrix2 = pd.read_excel(file_path2)
df_matrix2 = df_matrix2.drop('Unnamed: 0', axis=1)
# 将表格矩阵的值插入到DataFrame中
for i in range(df_matrix1.shape[0]):
    for j in range(df_matrix1.shape[1]):
        cell_value = df_matrix1.iloc[i, j]
        if pd.notnull(cell_value):
            paths = cell_value.split(',')
            if len(paths) == 1:
                df1.loc[df1.index[i * 8 + j], 'path1'] = str(paths[0])
            else:
                df1.loc[df1.index[i * 8 + j], 'path1'] = str(paths[0])
                if len(paths) > 1:
                    df1.loc[df1.index[i * 8 + j], 'path2'] = str(paths[1])
for i in range(df_matrix2.shape[0]):
    for j in range(df_matrix2.shape[1]):
        cell_value = df_matrix2.iloc[i, j]
        if pd.notnull(cell_value):
            paths = cell_value.split(',')
            if len(paths) == 1:
                df2.loc[df2.index[i * 8 + j], 'pathlink1'] = str(paths[0])
            else:
                df2.loc[df2.index[i * 8 + j], 'pathlink1'] = str(paths[0])
                if len(paths) > 1:
                    df2.loc[df2.index[i * 8 + j], 'pathlink2'] = str(paths[1])
df1 = pd.DataFrame(df1, dtype=str)
df2 = pd.DataFrame(df2, dtype=str)


print(df1)
print(df2)
# 初始化存储结果的数组
sequential_vehicle_counts = []
# 初始化存储结果的数组
vehicle_id_counts = [0] * 48
vehicle_id_sequences = []
# 读取CSV文件
esti_x = pd.read_csv(r"../渗透率/p=0.5/估计流量.csv")

# 定义Link ID的范围
link_ids_cro1 = [17, 19, 18, 20]
link_ids_cro2 = [26, 25, 24, 23]
link_ids_cro3 = [34, 33, 32, 31]
link_ids_cro4 = [42, 41, 40, 39]

# 初始化矩阵
cro1 = np.zeros((4, 3), dtype=int)
cro2 = np.zeros((4, 3), dtype=int)
cro3 = np.zeros((4, 3), dtype=int)
cro4 = np.zeros((4, 3), dtype=int)

# 填充矩阵
def fill_matrix(matrix, link_ids):
    for i, link_id in enumerate(link_ids):
        for j in range(1, 4):  # Lane ID从1到3
            value = esti_x[(esti_x['Link ID'] == link_id) & (esti_x['Lane ID'] == j)]['estimate_flow'].values[0]
            matrix[i, j-1] = value

fill_matrix(cro1, link_ids_cro1)
fill_matrix(cro2, link_ids_cro2)
fill_matrix(cro3, link_ids_cro3)
fill_matrix(cro4, link_ids_cro4)
# 将4个矩阵按行连接成一个12*3的矩阵
combined_matrix = np.vstack((cro1, cro2, cro3, cro4))
# 将12*3的矩阵展平为一个48*1的矩阵
Xi = combined_matrix.flatten()

# 遍历LinkID序列的DataFrame中的所有非零值
for index, row in df2.iterrows():
    for value in row:
        if value != '0':
            link_ids = str(value).split('-')
            # print(link_ids)
            # 初始化存储LinkID序列中的VehicleID的集合
            vehicle_ids = set()
            # 遍历LinkID序列
            for link_id in link_ids:
                specified_data = sampled_data[sampled_data['Link ID'] == int(link_id)]
                if not vehicle_ids:
                    vehicle_ids.update(specified_data['Vehicle ID'].unique())
                else:
                    vehicle_ids.intersection_update(specified_data['Vehicle ID'].unique())

                # 检查是否存在于整个LinkID序列的所有Link上
            if len(vehicle_ids) > 0:
                sequential_vehicle_counts.append(len(vehicle_ids))
                vehicle_id_sequences.append(value)
                # 将计算得到的值添加到df2中的第三列或第四列
                if len(sequential_vehicle_counts) > 0:
                    if value == row['pathlink1']:
                        df2.loc[index, 'NewCol1'] = sequential_vehicle_counts[-1]
                    elif value == row['pathlink2']:
                        df2.loc[index, 'NewCol2'] = sequential_vehicle_counts[-1]
# 输出带有新列的DataFrame df2
# 将 DataFrame 中的 NaN 值替换为 0
df2.fillna(0, inplace=True)
# 组合两个DataFrame的列
combined_data = pd.concat([df1, df2], axis=1)
# 创建DataFrame
df = pd.DataFrame(combined_data)
# print(df)

# 初始化新的DataFrame
new_df = pd.DataFrame(columns=['I', 'K', 'Q'])

# 用于记录每个元素出现的行号
element_rows = {}
# 遍历DataFrame
for index, row in df.iterrows():
    # 处理path1
    if row['path1'] != '0':
        elements = row['path1'].split('-')
        for element in elements:
            if element not in element_rows:
                element_rows[element] = []
            element_rows[element].append(index)

    # 处理path2
    if row['path2'] != '0':
        elements = row['path2'].split('-')
        for element in elements:
            if element not in element_rows:
                element_rows[element] = []
            element_rows[element].append(index)

# 根据记录的行号填充新的DataFrame
for element, indexes in element_rows.items():
    for idx in indexes:
        # 确定K列的值
        if idx in df[df['path1'].str.contains(element, na=False, regex=False)].index:
            link = 'pathlink1'
            col = 'NewCol1'
        else:
            link = 'pathlink2'
            col = 'NewCol2'

        # 使用 concat 来添加行
        new_df = pd.concat([new_df, pd.DataFrame({
            'I': [element],
            'K': [df.at[idx, link]],
            'Q': [df.at[idx, col]]
        })], ignore_index=True)
pd.set_option('display.max_rows', None)
print(new_df)
yik = []
for i in range(48):
    I_value = i + 1  # I 的取值从 1 到 48
    # print(I_value)
    # print(str(I_value))
    I_sum_Y = new_df[new_df['I'] == str(I_value)]['Q'].sum()
    # print(I_sum_Y)
    for j in range(len(new_df)):
        if new_df.at[j, 'I'] == str(I_value):
            K_value = new_df.at[j, 'K']
            Y_value = new_df.at[j, 'Q']
            # 分子 nk
            nk = Y_value
            if I_sum_Y != 0:
                yik_value = (nk / I_sum_Y) * Xi[I_value-1]
                yik.append(yik_value)
print(yik)
print(len(yik))
# 创建一个空列表用于存放结果
new_list = []
# 遍历 DataFrame 按顺序逐行处理
for row in df1.itertuples(index=False):
    for value in row:
        if value != '0':
            split_values = value.split('-')
            new_list.append(len(split_values))
# 补充缺少的值为 0，直到新列表长度为 72
while len(new_list) < 72:
    new_list.append(0)
yk=[]
# 计算 yk 列表的值
for i in range(72):
    if i < len(yik) and i < len(new_list):
        if new_list[i] != 0:
            yk.append(yik[i] / new_list[i])
        else:
            yk.append(0)
    else:
        yk.append(0)
yk_matrix = np.array([arr[0] for arr in yk])
# print(yk_matrix)
# 创建48个长度为72的列表，全部初始化为0
lists = [[0] * 72 for _ in range(48)]
for i in range(48):
    count = 0  # 重置count，准备处理下一个i
    for index,row in df1.iterrows():
        for value in row:
            if value != '0':  # 检查是否值不等于字符串'0'
                count=count+1
                values = value.split('-')  # 使用'-'分割字符串
                for v in values:
                    # print(v,"huanhang")
                    if str(i + 1) == v:
                        lists[i][count-1]=1
        # print("(",count,")")
        continue
    if count==72:
        continue

A = np.array(lists) # 路径和路段流向的关联系数
