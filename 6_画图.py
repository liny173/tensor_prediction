import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取CSV文件
real_flow_df = pd.read_csv('真实流量.csv')
estimate_flow_df = pd.read_csv('数据上传间隔/interval=5/估计流量.csv')
png_save = '数据上传间隔/interval=5/FGD.png'

# 计算 MAPE 值
def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100


# 合并两个DataFrame，根据Link ID和Lane ID进行合并
merged_df = pd.merge(real_flow_df, estimate_flow_df, on=['Link ID', 'Lane ID'], suffixes=('_real', '_estimate'))

# 创建一个新列，用于直方图的横轴
merged_df['Link ID_Lane ID'] = merged_df['Link ID'].astype(str) + '_' + merged_df['Lane ID'].astype(str)


# 计算误差
mape_value = calculate_mape(merged_df['real_flow'], merged_df['estimate_flow'])
sum_mape = calculate_mape(sum(merged_df['real_flow']), sum(merged_df['estimate_flow']))

# 定义柱的宽度
bar_width = 0.35

# 定义每个柱的位置
index = np.arange(len(merged_df['Link ID_Lane ID']))

# 创建柱状图
plt.figure(figsize=(12, 6))
plt.xticks(rotation=45)
plt.bar(index, merged_df['estimate_flow'], bar_width, label='Estimated flow', color='b')  # 估计流量
plt.bar(index + bar_width, merged_df['real_flow'], bar_width, label='Actual flow', color='g')  # 实际流量

# 设置x轴标签和标题
plt.xlabel('x_y values')
plt.ylabel('Flow')
plt.title('Comparison of Estimated and Actual Flow')

# 设置x轴的刻度和标签
plt.xticks(index + bar_width / 2, merged_df['Link ID_Lane ID'], rotation=45)

# 添加图例
plt.legend()

# 标记 MAPE 值在图中间上方
plt.text(0.5, 0.8, f'MAPE: {mape_value:.2f}%', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.5))
plt.text(0.5, 0.6, f'SUM_MAPE: {sum_mape:.2f}%', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.5))

# 设置图形的标题和标签
plt.title('Histogram of Real and Estimated Flows')
plt.xlabel('Lane ID_Link ID')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # 旋转x轴标签，以便更好地显示

# 显示图例
plt.legend()

# 显示图形
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.savefig(png_save)
plt.show()
