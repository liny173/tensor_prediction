file_save = "数据上传间隔/interval=5/受控流量.csv"
import pandas as pd

# 读取CSV文件
df = pd.read_csv('数据上传间隔/interval=5/est_Q_FGD.csv')

# 定义要计算中位数的列
columns_to_median = ['est_flow_1', 'est_flow_2', 'est_flow_3', 'est_flow_4', 'est_flow_5']

# 计算中位数，并将结果存储在新列estimate_flow中
df['estimate_flow'] = df[columns_to_median].median(axis=1)

# 创建新的DataFrame，只包含Link ID，Lane ID和estimate_flow
new_df = df[['Link ID', 'Lane ID', 'estimate_flow']]

# 保存新的DataFrame到CSV文件
new_df.to_csv(file_save, index=False)

# 打印新的DataFrame查看结果
print(new_df)
