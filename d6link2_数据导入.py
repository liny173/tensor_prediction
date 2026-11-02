import pandas as pd

df = pd.read_csv(r'D:\01\traffic\tensor prediction\data\20181029_d6_1000_1030.csv')


# 定义一个函数来处理每一行数据
def split_data(row):
    # 按分号拆分每一行数据
    values = row['track_id; type; traveled_d; avg_speed; lat; lon; speed; lon_acc; lat_acc; time'].split(';')
    values = [v.strip() for v in values if v]  # 去除空格和空字符串

    # 提取第一个和第二个数据
    first_data = values[0]
    second_data = values[1]

    # 从第五个数据开始，每六个为一组
    grouped_data = []
    for i in range(4, len(values), 6):
        if i + 5 < len(values):  # 确保组内有六个数据
            new_row = [first_data, second_data] + values[i:i + 6]
            grouped_data.append(new_row)

    return grouped_data


# 处理所有行
expanded_rows = []
for _, row in df.iterrows():
    expanded_rows.extend(split_data(row))

# 将结果转换为新的DataFrame
new_df = pd.DataFrame(expanded_rows,
                      columns=['ID', 'Type', 'Latitude', 'Longitude', 'Speed', 'Lat_accuracy', 'Lon_accuracy', 'Time'])
new_df['Latitude'] = pd.to_numeric(new_df['Latitude'], errors='coerce')
new_df['Longitude'] = pd.to_numeric(new_df['Longitude'], errors='coerce')

# print(new_df)
latitude_min = 37.985825  # 最小纬度
latitude_max = 37.985840  # 最大纬度
longitude_min = 23.729753  # 最小经度
longitude_max = 23.729812  # 最大经度

# 过滤数据
filtered_df = new_df[(new_df['Latitude'] >= latitude_min) & (new_df['Latitude'] <= latitude_max) &
                     (new_df['Longitude'] >= longitude_min) & (new_df['Longitude'] <= longitude_max) & (
                             new_df['Type'] != 'Motorcycle')]

# 查看新的 DataFrame
print(filtered_df)
filtered_df.to_csv('D:\\01\\traffic\\tensor prediction\data_4\预处理数据_20181029_d6_1000_1030_22.csv', index=False)
