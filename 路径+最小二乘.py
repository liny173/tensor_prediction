import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

chunk_size = 1000000  # 每次读取的行数
chunks = []
for chunk in pd.read_csv(r"D:\01\traffic\tensor prediction\路径估计值+最小二乘\渗透率p\抽样轨迹数据\抽样轨迹数据\抽样轨迹数据_0.1.csv", chunksize=chunk_size):
    chunks.append(chunk)
# 合并所有块
sampled_data = pd.concat(chunks, ignore_index=True)

# 创建行名列表
# 列表, 内容['1-1']
rows = [f"{i}-{j}" for i in range(1, 9) for j in range(1, 9)]
# 字典
data1 = {'path1': [0] * 64, 'path2': [0] * 64}
data2 = {'pathlink1': [0] * 64, 'pathlink2': [0] * 64}
# Dataframe
# 内容 1-1: path1(0) path2(0)
#     1-2: path1(0) path2(0)
df1 = pd.DataFrame(data1, index=rows)
df2 = pd.DataFrame(data2, index=rows)

file_path1 = r"D:\01\traffic\tensor prediction\流向流量估计\路径流量\流向路径表格.xlsx"
file_path2 = r"D:\01\traffic\tensor prediction\流向流量估计\路径流量\路段路径表格.xlsx"
df_matrix1 = pd.read_excel(file_path1)
df_matrix1 = df_matrix1.drop('Unnamed: 0', axis=1)
df_matrix2 = pd.read_excel(file_path2)
df_matrix2 = df_matrix2.drop('Unnamed: 0', axis=1)

# df1是流向路径
# df2是车道路径
# 将表格矩阵的值插入到DataFrame中
for i in range(df_matrix1.shape[0]):  # 8行
    for j in range(df_matrix1.shape[1]):  # 8列
        cell_value = df_matrix1.iloc[i, j]
        # 非空
        if pd.notnull(cell_value):
            # 字符串切片
            paths = cell_value.split(',')  # 不只一条路径, 最多为两条路径, 路径由','号分隔
            if len(paths) == 1:
                df1.loc[df1.index[i * 8 + j], 'path1'] = str(paths[0])
            else:
                df1.loc[df1.index[i * 8 + j], 'path1'] = str(paths[0])
                if len(paths) > 1:
                    df1.loc[df1.index[i * 8 + j], 'path2'] = str(paths[1])

# 处理方式同上
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

# 初始化存储结果的数组
sequential_vehicle_counts = []

# 初始化存储结果的数组
vehicle_id_counts = [0] * 48
vehicle_id_sequences = []

# 拿到先验的流向流量数据
# 读取CSV文件
esti_x = pd.read_csv(r"D:\01\traffic\tensor prediction\路径估计值+最小二乘\渗透率p\估计流量数据\估计流量数据\p=0.1\估计流量.csv")
# 定义Link ID的范围
# 这个是依据仿真的数据
link_ids_cro1 = [17, 19, 18, 20]
link_ids_cro2 = [26, 25, 24, 23]
link_ids_cro3 = [34, 33, 32, 31]
link_ids_cro4 = [42, 41, 40, 39]

# 初始化矩阵
# 4个交叉口, 每个交叉口4个进口道, 每个进口道左直右3个车道
cro1 = np.zeros((4, 3), dtype=int)
cro2 = np.zeros((4, 3), dtype=int)
cro3 = np.zeros((4, 3), dtype=int)
cro4 = np.zeros((4, 3), dtype=int)


# 填充矩阵
def fill_matrix(matrix, link_ids):
    for i, link_id in enumerate(link_ids):
        for j in range(3, 0, -1):  # Lane ID从3到1
            value = esti_x[(esti_x['Link ID'] == link_id) & (esti_x['Lane ID'] == j)]['estimate_flow'].values[0]
            matrix[i, 3-j] = value


fill_matrix(cro1, link_ids_cro1)
fill_matrix(cro2, link_ids_cro2)
fill_matrix(cro3, link_ids_cro3)
fill_matrix(cro4, link_ids_cro4)
# 将4个矩阵按行连接成一个12*3的矩阵
combined_matrix = np.vstack((cro1, cro2, cro3, cro4))
# 将12*3的矩阵展平为一个48*1的矩阵
Xi = combined_matrix.flatten()

# 获得抽样轨迹数据nk
for index, row in df2.iterrows():
    for value in row:
        if value != '0':
            # link_ids就是17-26-10拆开后
            link_ids = str(value).split('-')
            # print(link_ids)
            # 初始化存储LinkID序列中的VehicleID的集合
            vehicle_ids = set()
            # 遍历LinkID序列
            for link_id in link_ids:
                specified_data = sampled_data[sampled_data['Link ID'] == int(link_id)]  # 从抽样数据中
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

# 获得路径流量的真实值-从抽样轨迹数据中获得
# Y_list = []
# for index, row in df.iterrows():
#     if row['pathlink1'] != '0':
#         Y_list.append(row['NewCol1'])
#     if row['pathlink2'] != '0':
#         Y_list.append(row['NewCol2'])
#     if len(Y_list) >= 72:
#         break
# # 将列表转换为NumPy数组
# result_array = np.array(Y_list)
# # 将数组形状改为(72, 1)的矩阵
# Y_real = result_array.reshape(72, 1)
# print(Y_real)
from sklearn.preprocessing import StandardScaler
# 得到真实的路径流量数据
Y_real = np.loadtxt(fr"E:\大三下\科创大创\实验室\Y_real_data.csv")
# print(Y_real)
# Y_real = np.where(Y_real < 20, np.random.uniform(20, 80), Y_real)  # 小于 20 的值随机映射到 75-151 之间
# Y_real = np.where(Y_real > 350, np.random.uniform(20, 80), Y_real)  # 大于 350 的值随机映射到 75-151 之间
# print(Y_real)
from sklearn.preprocessing import StandardScaler
# 初始化新的DataFrame
new_df = pd.DataFrame(columns=['I', 'K', 'Q'])

# 用于记录每个元素出现的行号
element_rows = {}
# 遍历DataFrame df, 得到了路径流量的DataFrame
for index, row in df.iterrows():  # index是'1-1'
    # 处理path1
    if row['path1'] != '0':
        elements = row['path1'].split('-')  # 路段
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
# 最终得到: 经过的流向路段od对, 例:
# '1': ['1-2'], 意思是经过流向路径1, od对为1-2

# 根据记录的行号填充新的DataFrame
for element, indexes in element_rows.items():
    for idx in indexes:
        # 确定K列的值
        if idx in df[df['path1'].str.contains(element, na=False, regex=False)].index:  # str的contains()方法, 用于检查包含指定的子字符串
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
# 最终得到, 流向+路径+流量, 例:
# I(1) K(17-4) Q(11), 流向1, 路径为17-4, 流量为11

# 拿到关联矩阵A
# 创建48个长度为72的列表，全部初始化为0
# 将48个流向, 与72条路径关联起来
lists = [[0] * 72 for _ in range(48)]
for i in range(48):
    count = 0  # 重置count，准备处理下一个i
    for index, row in df1.iterrows():
        for value in row:
            if value != '0':  # 检查是否值不等于字符串'0'
                count += 1
                values = value.split('-')  # 使用'-'分割字符串
                for v in values:
                    # print(v,"huanhang")
                    if str(i + 1) == v:
                        lists[i][count-1] = 1
        # print("(",count,")")
        continue
    if count == 72:
        continue


# 设置Pandas显示选项，使DataFrame的内容不被省略
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# 设置NumPy的显示选项，使矩阵的内容不被省略
np.set_printoptions(threshold=np.inf)
A = np.array(lists)  # 路径和路段流向的关联系数

# print(df)
non_zero_count = np.count_nonzero(A)
# print(non_zero_count)
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
# 初始化一个48*72的全零矩阵B
B = np.zeros((48, 72))
# 遍历矩阵A，将yik中的值填充到A的非零值处成为B
non_zero_indices = np.argwhere(A != 0)  # 获取A中非零值的索引

for idx, (row, col) in enumerate(non_zero_indices):
    if idx < len(yik):
        B[row, col] = yik[idx]
# print(A)
# print(B)
# print(yik)
# 计算矩阵B每列非零值的平均值并处理分母为0的情况
column_sums = np.sum(B, axis=0)  # 对每一列求和
non_zero_counts = np.count_nonzero(B, axis=0)  # 每列非零值数量

# 处理分母为0的情况，将分母为0的位置对应的平均值设置为0
mask = non_zero_counts > 0
yk = np.zeros(B.shape[1])  # 初始化平均值矩阵
yk[mask] = column_sums[mask] / non_zero_counts[mask]
# 真实流向流量数据
realcro1 = [[109, 360, 111],
            [126, 323, 164],
            [107, 292, 103],
            [128, 275, 143]]
realcro2 = [[282, 202, 98],
            [286, 203, 113],
            [241, 160, 71],
            [251, 215, 103]]
realcro3 = [[314, 225, 117],
            [271, 176, 89],
            [311, 207, 110],
            [86, 248, 105]]
realcro4 = [[111, 191, 281],
            [97, 321, 112],
            [150, 432, 157],
            [102, 348, 118]]

# 将4个矩阵按行连接成一个12*3的矩阵
combined_matrix2 = np.vstack((realcro1, realcro2, realcro3, realcro4))
# 将12*3的矩阵展平为一个48*1的矩阵
Xi2 = combined_matrix.flatten()

x_init = np.random.uniform(low=50, high=230, size=(48, 1))

# 生成一个72x1的矩阵，元素值在20到180之间
y_init = np.random.uniform(low=0, high=80, size=(72, 1))


def compute_gradient(X, Y, omega1, omega2, X_hat, Y_hat):
    gradient_x = 2 * omega1 * (X - X_hat) / np.sum(X_hat ** 2)
    gradient_y = 2 * omega2 * (Y - Y_hat) / np.sum(Y_hat ** 2)
    return gradient_x, gradient_y  # (72,1)


def optimize_adam(X_init, Y_init, X_hat, Y_hat, omega1, omega2,
                  A, alpha,beta1=0.1, beta2=0.5):
    X = X_init
    Y = Y_init
    t = 0
    epsilon = 1e-5  # 避免分母为0
    learning_rate = 0.001  # 初始学习率
    m_x = 0  # Adam算法中的梯度一阶矩估计
    v_x = 0  # Adam算法中的梯度二阶矩估计
    m_y = 0
    v_y = 0

    while t < 65000:
        gradient_x, gradient_y = compute_gradient(X, Y, omega1, omega2, X_hat, Y_hat)

        norm_gradient_x = np.linalg.norm(gradient_x)
        norm_gradient_y = np.linalg.norm(gradient_y)
        print("Iteration:", t)
        print("Gradient X:", norm_gradient_x)
        print("Gradient Y:", norm_gradient_y)
        m_x = beta1 * m_x + (1 - beta1) * gradient_x
        v_x = beta2 * v_x + (1 - beta2) * (gradient_x ** 2)
        m_x_hat = m_x / (1 - beta1 ** (t + 1))
        v_x_hat = v_x / (1 - beta2 ** (t + 1))

        m_y = beta1 * m_y + (1 - beta1) * gradient_y
        v_y = beta2 * v_y + (1 - beta2) * (gradient_y ** 2)
        m_y_hat = m_y / (1 - beta1 ** (t + 1))
        v_y_hat = v_y / (1 - beta2 ** (t + 1))

        X -= learning_rate * m_x_hat / (np.sqrt(v_x_hat) + epsilon)
        Y -= learning_rate * m_y_hat / (np.sqrt(v_y_hat) + epsilon)
        Y = np.maximum(Y, 0) + alpha

        X_updated = np.dot(A, Y)
        Y_updated = np.maximum(Y - alpha, 0)

        if np.linalg.norm(gradient_x) < epsilon and np.linalg.norm(gradient_y) < epsilon:
            break

        X = X_updated
        Y = Y_updated
        t += 1

    print("Total Iterations:", t)
    return X, Y


X_hat = np.array(Xi)  # 流向流量先验值
X_hat = X_hat.reshape((48, 1))
Y_hat = np.array(yk)  # 路径流量先验值
Y_hat = Y_hat.reshape((72, 1))
# X_init = np.array(Xi2)
# X_init=X_init.reshape((48,1))
# Y_init = np.array(Y_real)
# Y_init=Y_init.reshape((72,1))
omega1 = 0.1  # 误差项1的权重系数
omega2 = 0.001  # 误差项2的权重系数
alpha = 0.099  # 学习率
# 使用Adam算法进行优化
# X_optimal, Y_optimal = optimize_adam(x_init, y_init, X_hat, Y_hat, omega1, omega2,
#                                     A,alpha)
# # 直接保存 X_optimal 到当前工作目录下的 X_optimal.txt 文件
# np.savetxt('X_optimal-pw=0.01.txt', X_optimal)
#
# # 直接保存 Y_optimal 到当前工作目录下的 Y_optimal.txt 文件
# np.savetxt('Y_optimal-pw=0.01.txt', Y_optimal)

# 假设渗透率列表
pw = [0.01, 0.1, 1, 10, 100]

# 初始化存储误差的列表
mae_list_X = []
wmape_list_X = []
rmse_list_X = []
mae_list_Y = []
wmape_list_Y = []
rmse_list_Y = []


# 计算平均绝对误差（IRMAE）：
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# 计算加权平均百分误差（IWMAPE）：
def weighted_mean_absolute_percentage_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100


# 计算均方根误差（IRMSE）：
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


for w in pw:
    Y_optimal = np.loadtxt(fr'E:\大三下\科创大创\实验室\权重系数\Y_optimal-pw={w}.txt')
    # 计算路径流量的MAE
    MAE_Y = mean_absolute_error(Y_real, Y_optimal)
    # 计算流向流量的WMAPE
    WMAPE_Y = weighted_mean_absolute_percentage_error(Y_real, Y_optimal)
    # 计算流向流量的RMSE
    RMSE_Y = root_mean_squared_error(Y_real, Y_optimal)
#
#     # 将结果添加到列表中
#     # mae_list_X.append(MAE_X)
#     # wmape_list_X.append(WMAPE_X)
#     # rmse_list_X.append(RMSE_X)
    mae_list_Y.append(MAE_Y)
    wmape_list_Y.append(WMAPE_Y)
    rmse_list_Y.append(RMSE_Y)

# 绘制图表
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制柱状图
bar_width = 0.1
index = np.arange(len(pw))

bar1 = ax1.bar(index, mae_list_Y, bar_width, label='MAE', color='skyblue')  # color='51b1b7'
bar2 = ax1.bar(index + bar_width, rmse_list_Y, bar_width, label='RMSE', color='orange')  # color='e07b54'

# 设置左侧y轴标签
ax1.set_xlabel('Weighting Coefficient')
ax1.set_ylabel('MAE or RMSE')
ax1.set_xticks(index + bar_width)
ax1.set_xticklabels(pw)
ax1.legend(loc='upper left')

# 创建第二个y轴并绘制折线图
ax2 = ax1.twinx()
line = ax2.plot(index + bar_width / 2, wmape_list_Y, 'o-', label='WMAPE', color='green')  # color='9ec3aa'
ax2.set_ylabel('WMAPE%')

# 添加图例
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

# 显示图表
plt.title("Comparison of Data Precision for Different Weighting Coefficient")
plt.show()

# 画第二种图
# 数据
x = list(range(1, 49))  # 横轴数据，表示矩阵的元素位置
optimalx = np.loadtxt(fr"E:\大三下\科创大创\实验室\权重系数\X_optimal-pw=1.txt")
# 创建图像
plt.figure(figsize=(10, 8))

# 第一张图
plt.subplot(2, 1, 1)
plt.plot(x, Xi2, 'o-', label='real')
plt.plot(x, optimalx, 's-', label='estimate')
plt.title('direction')
plt.xlabel('location')
plt.ylabel('Q')
plt.legend()

# 第二张图
y = list(range(1, 73))  # 横轴数据，表示矩阵的元素位置
optimaly = np.loadtxt(fr"E:\大三下\科创大创\实验室\权重系数\Y_optimal-pw=1.txt")
plt.subplot(2, 1, 2)
plt.plot(y, Y_real, 'o-', label='real')
plt.plot(y, optimaly, 's-', label='estimate')
plt.title('path')
plt.xlabel('location')
plt.ylabel('Q')
plt.legend()

plt.tight_layout()
plt.show()