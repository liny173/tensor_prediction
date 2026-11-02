import numpy as np
import pandas as pd
import tensorly as tl
from scipy.optimize import minimize_scalar

# 读取 CSV 文件并加载数据
# 逐块读取 CSV 文件
chunk_size = 1000000  # 每次读取的行数

# 存入列表
chunks = []

for chunk in pd.read_csv('抽样轨迹数据_间隔_5.csv', chunksize=chunk_size):
    chunks.append(chunk)

# 合并所有块
sampled_data = pd.concat(chunks, ignore_index=True)
file_save = '数据上传间隔/interval=5/est_Q_FGD.csv'

# 参数C
C = 103

# 读入路段数据
link_lane_data = pd.read_csv('路段信息整合.csv')

# 创建估计流量的列表
all_Q_est = []

# 判断车辆的状态，并计算车辆的预计相对到达时间
def calculate_rpat(data, pos, stop_lines):

    # 结果的列表
    results = []

    # group by Vehicle ID
    grouped = data.groupby('Vehicle ID')

    for vehicle_id, group in grouped:  # 按照车辆id分组

        group = group.sort_values('Timestep')  # 根据时间排序
        group['is_stopping'] = group['Speed'] == 0  # 划分是否停车
        stop_times = group[group['is_stopping'] == True]
        move_times = group[group['is_stopping'] == False]

        # 保证车辆信息的有效性: not move_times.empty and not stop_times.empty
        # 情况一: 排队(移动时间在停止时间之后)
        if (not move_times.empty and not stop_times.empty
                and move_times['Timestep'].iloc[-1] > stop_times['Timestep'].iloc[-1]):
            if pos == 'X':
                stop_pos = abs(stop_times['Front X'].iloc[0] - stop_lines)
            if pos == 'Y':
                stop_pos = abs(stop_times['Front Y'].iloc[0] - stop_lines)

            projected_arrival_time = stop_times['Timestep'].iloc[0] + stop_pos / move_times['Speed'].iloc[0]  # 获取预计到达时间
            departure_time = move_times['Timestep'].iloc[-1]  # 获取离开时间

            results.append({
                'Vehicle ID': vehicle_id,
                'stop_pos': stop_pos,
                'projected_arrival_time': (projected_arrival_time) % C,
                'departure_time': departure_time % C,
                'Cycle': ((departure_time) / C).astype(int) + 1,
                'type': 'queue'
            })

        # 情况二: 畅行
        if stop_times.empty and not move_times.empty:
            departure_time = move_times['Timestep'].iloc[-1]

            results.append({
                'Vehicle ID': vehicle_id,
                'stop_pos': 0,
                'projected_arrival_time': (departure_time) % C,
                'departure_time': departure_time % C,
                'Cycle': ((departure_time) / C).astype(int) + 1,
                'type': 'free'
            })

    results_df = pd.DataFrame(results)
    # results_df = results_df.sort_values('departure_time')
    return results_df


"""
论文
A tensor decomposition method for cycle-based traffic volume 
estimation using sampled vehicle trajectories
a feature matrix can be constructed,  corresponding to the features (volume, sampled vehicle 
count, and observation duration)
"""
def construct_feature_matrix(data):
    matrices = []
    for (v_bef, v_aft, n_bef, n_aft, t_bef, t_aft) in data:
        matrix = np.array([[v_bef, v_aft], [n_bef, n_aft], [t_bef, t_aft]])
        matrices.append(matrix)

    return np.array(matrices)

# 初始化张量和参数
def initialize(tensor_shape, rank):

    # the core tensor
    core = np.random.randn(rank[0], rank[1], rank[2])  # core的shape是(3,2,2)

    # the factor matrices
    A = np.random.randn(tensor_shape[0], rank[0])  # A的shape是(3,3)
    B = np.random.randn(tensor_shape[1], rank[1])  # B的shape是(2,2)
    C = np.random.randn(tensor_shape[2], rank[2])  # C的shape是(4,2)

    return core, A, B, C

# 计算预测张量
def predict(core, A, B, C):

    # 计算模态积
    M_list = [A, B, C]
    mode_list = [0, 1, 2]

    # 函数multi_mode_dot()，计算得到一个张量和多个矩阵或向量在多个模态(张量的阶)上的 n 众态积
    predict_tensor = tl.tenalg.multi_mode_dot(core, M_list, mode_list)
    # print(predict_tensor.shape)
    return predict_tensor


# 定义目标函数和正则化项
"""
W is a non-negative weight tensor the same size as core, which 
is used to indicate where known values exist, 0 or 1
"""
def objective(tensor, core, A, B, C, W, lambda_reg):

    # 计算预测张量
    reconstruction = predict(core, A, B, C)

    # 误差项
    error = W * (tensor - reconstruction)
    regularization = lambda_reg * (np.linalg.norm(core) ** 2 + np.linalg.norm(A) ** 2 +
                                   np.linalg.norm(B) ** 2 + np.linalg.norm(C) ** 2)

    # 论文第六页，Eq.11
    return np.linalg.norm(error) ** 2 + regularization


# # 梯度更新函数
# def update_parameters(tensor, core, A, B, C, W, beta, lambda_reg):
#     tensor_shape = tensor.shape  # tensor.shape = (3,2,4)
#     # print(C.shape)
#     for i in range(tensor_shape[0]):
#         for j in range(tensor_shape[1]):
#             for k in range(tensor_shape[2]):
#                 if W[i, j, k] != 0:
#                     beta += 1
#                     alpha = 1 / np.sqrt(beta)
#                     # X_hat = np.tensordot(np.tensordot(np.tensordot(core, A[i, :], axes=[0, 0]), B[j, :], axes=[0, 0]),
#                     #                      C[k, :], axes=[0, 0])
#                     # error = X_hat - tensor[i, j, k]
#
#                     # X_hat = np.tensordot(C[k, :], np.tensordot(B[j, :],
#                     #                                            np.tensordot(A[i, :], core,
#                     #                                                         axes=(0, 0)), axes=(0, 0)),
#                     #                      axes=(0, 0))  # Scalar
#                     X_hat = tl.tenalg.multi_mode_dot(core, [A[i, :], B[j, :], C[k, :]], [0, 1, 2])
#                     error = X_hat - tensor[i, j, k]
#                     # print(X_hat)
#                     # A[i, :] -= alpha * (
#                     #         lambda_reg * A[i, :] + error * tl.tenalg.mode_dot(tl.tenalg.mode_dot(core, B[j, :], 1),
#                     #                                                           C[k, :], 2))
#
#                     # A[i, :] -= alpha * (
#                     #         lambda_reg * A[i, :] + error * np.tensordot(np.tensordot(core, B[j, :], axes=[1, 0]),
#                     #                                                     C[k, :], axes=[1, 0]))
#                     A[i, :] -= alpha * (
#                             lambda_reg * A[i, :] + error * tl.tenalg.multi_mode_dot(core, [B[j, :], C[k, :]],
#                                                                                     [1, 2]))
#                     # B[j, :] -= alpha * (
#                     #         lambda_reg * B[j, :] + error * np.tensordot(np.tensordot(core, A[i, :], axes=[0, 0]),
#                     #                                                     C[k, :], axes=[2, 0]))
#                     B[j, :] -= alpha * (
#                             lambda_reg * B[j, :] + error * tl.tenalg.multi_mode_dot(core, [A[i, :], C[k, :]],
#                                                                                     [0, 2]))
#                     # C[k, :] -= alpha * (
#                     #         lambda_reg * C[k, :] + error * np.tensordot(np.tensordot(core, A[i, :], axes=[0, 0]),
#                     #                                                     B[j, :], axes=[1, 0]))
#                     C[k, :] -= alpha * (
#                             lambda_reg * C[k, :] + error * tl.tenalg.multi_mode_dot(core, [A[i, :], B[j, :]],
#                                                                                     [0, 1]))
#
#                     core -= alpha * (
#                             lambda_reg * core + error * np.outer(A[i, :], np.outer(B[j, :], C[k, :])).reshape(
#                         core.shape))
#                     # core -= alpha * (lambda_reg * core + error * np.kron(np.kron(A[i, :], B[j, :]), C[k, :]))
#
#     return beta, core, A, B, C
#
#
# # 梯度下降张量分解算法
# def tensor_decomposition(tensor, W, epsilon=0.001, lambda_reg=0.01, max_iter=1000):
#     tensor_shape = tensor.shape
#     rank = [3, 2, 4]
#     core, A, B, C = initialize(tensor_shape, rank)
#     # print(core, A, B, C)
#     theta = objective(tensor, core, A, B, C, W, lambda_reg)
#     delta_theta = float('inf')
#     iteration = 0
#     beta = 10 ** 8
#     while delta_theta > epsilon and iteration < max_iter:
#         beta, core, A, B, C = update_parameters(tensor, core, A, B, C, W, beta, lambda_reg)
#         new_theta = objective(tensor, core, A, B, C, W, lambda_reg)
#         delta_theta = abs(theta - new_theta)
#         theta = new_theta
#         iteration += 1
#         # print(f"Iteration {iteration}, Objective: {theta}, Delta: {delta_theta}")
#     # print(core, A, B, C)
#     return predict(core, A, B, C)


# 考虑受控流向


def compute_gradient(tensor, core, A, B, C, W, lambda_reg):
    tensor_shape = tensor.shape
    grad_A = np.zeros_like(A)
    grad_B = np.zeros_like(B)
    grad_C = np.zeros_like(C)
    grad_core = np.zeros_like(core)

    # 张量第一阶
    for i in range(tensor_shape[0]):
        # 张量第二阶
        for j in range(tensor_shape[1]):
            # 张量第三阶
            for k in range(tensor_shape[2]):

                if W[i, j, k] != 0:    # indicate values existing
                    X_hat = tl.tenalg.multi_mode_dot(core, [A[i, :], B[j, :], C[k, :]], [0, 1, 2])
                    error = X_hat - tensor[i, j, k]
                    grad_A[i, :] += error * tl.tenalg.multi_mode_dot(core, [B[j, :], C[k, :]], [1, 2])
                    grad_B[j, :] += error * tl.tenalg.multi_mode_dot(core, [A[i, :], C[k, :]], [0, 2])
                    grad_C[k, :] += error * tl.tenalg.multi_mode_dot(core, [A[i, :], B[j, :]], [0, 1])
                    grad_core += error * np.outer(A[i, :], np.outer(B[j, :], C[k, :])).reshape(core.shape)

    grad_A += lambda_reg * A
    grad_B += lambda_reg * B
    grad_C += lambda_reg * C
    grad_core += lambda_reg * core

    return grad_core, grad_A, grad_B, grad_C


def update_parameters(tensor, core, A, B, C, W, lambda_reg):
    # 梯度计算
    grad_core, grad_A, grad_B, grad_C = compute_gradient(tensor, core, A, B, C, W, lambda_reg)

    def objective_alpha(alpha):
        new_core = core - alpha * grad_core
        new_A = A - alpha * grad_A
        new_B = B - alpha * grad_B
        new_C = C - alpha * grad_C

        return objective(tensor, new_core, new_A, new_B, new_C, W, lambda_reg)

    # 最小化
    res = minimize_scalar(objective_alpha)
    alpha = res.x

    core -= alpha * grad_core
    A -= alpha * grad_A
    B -= alpha * grad_B
    C -= alpha * grad_C

    return core, A, B, C


# 梯度下降张量分解算法
def tensor_decomposition(tensor, W, epsilon=0.001, lambda_reg=0.01, max_iter=1000):

    # 初始化
    tensor_shape = tensor.shape
    rank = [3, 2, 5]
    core, A, B, C = initialize(tensor_shape, rank)
    theta = objective(tensor, core, A, B, C, W, lambda_reg)
    delta_theta = float('inf')
    iteration = 0

    # 迭代寻优
    while delta_theta > epsilon and iteration < max_iter:
        core, A, B, C = update_parameters(tensor, core, A, B, C, W, lambda_reg)
        new_theta = objective(tensor, core, A, B, C, W, lambda_reg)
        delta_theta = abs(theta - new_theta)
        theta = new_theta
        iteration += 1
        # print(f"Iteration {iteration}, Objective: {theta}, Delta: {delta_theta}")

    return predict(core, A, B, C)


for index, row in link_lane_data.iterrows():    # pandas中的函数iterrrows()，遍历元组

    # 估计流量
    Q = []
    link_id = row['Link ID']
    lane_id = row['Lane ID']
    pos = row['Pos']
    stopline = row['Stopline']
    cycle_start = row['cycle_start']

    # 提取属性列
    origin_data = \
        sampled_data[
            (sampled_data['Link ID'] == link_id) & (sampled_data['Lane ID'] == lane_id)][
            ['Timestep', 'Vehicle ID', 'Link ID', 'Lane ID', 'Front X', 'Front Y', 'Speed']]

    # 筛选
    origin_data = origin_data[origin_data['Timestep'] > cycle_start]
    origin_data['Timestep'] = origin_data['Timestep'] - cycle_start

    data_pro = calculate_rpat(origin_data, pos, stopline)

    # 以周期分组
    data_group = data_pro.groupby('Cycle')

    # omega is the number of lanes for the subject movement
    # d_0 is the average spacing headway of queued vehicles
    d_0 = 7.5
    omega = 1

    cycle_flag = 1
    cycle_num = round((3600 - cycle_start) / C)
    data = [(0, 0, 0, 0, 0, 0)] * cycle_num

    for cycle, group in data_group:
        group = group.sort_values('departure_time')

        # 排队的车辆
        queue_list = group[group['type'] == 'queue']
        if not queue_list.empty:
            stop_last_index = queue_list['Vehicle ID'].iloc[-1]
            d_stop = queue_list[queue_list['Vehicle ID'] == stop_last_index]['stop_pos'].values[0]
            t_stop = queue_list[queue_list['Vehicle ID'] == stop_last_index]['projected_arrival_time'].values[0]
            last_stop_time = queue_list[queue_list['Vehicle ID'] == stop_last_index]['departure_time'].values[0]

            # 论文第三页，Eq.3
            v_bef = round(omega * d_stop / d_0) + 1

            v_aft = 0
            n_bef = group[group['departure_time'] <= last_stop_time]['Vehicle ID'].nunique()
            n_aft = group[group['departure_time'] > last_stop_time]['Vehicle ID'].nunique()

            t_bef = t_stop
            t_aft = C - t_bef
            data[int(cycle) - 1] = (v_bef, v_aft, n_bef, n_aft, t_bef, t_aft)

    data_array = np.array(data)
    zero_mask = np.all(data_array == 0, axis=1)

    non_zero_indices = np.where(~zero_mask)[0]
    if len(non_zero_indices) == 0:
        continue

    if zero_mask[0]:
        next_non_zero_index = non_zero_indices[0]
        data_array[0] = data_array[next_non_zero_index]

    for i in range(1, len(data_array)):

        # 论文第四页，To solve problem
        # replace the missing matrix with the matrix for the previous cycle
        # with queued sample vehicles as a substitute
        if zero_mask[i]:
            data_array[i] = data_array[i - 1]

    processed_data = [tuple(row) for row in data_array]

    X = construct_feature_matrix(processed_data)
    # 转置
    X = np.transpose(X, (1, 2, 0))
    # print(X)

    # 论文第四页，Fig.2(b)
    # 初始化omega权重矩阵
    W = np.ones((3, 2, cycle_num))
    W[0, 1, :] = 0
    cal_list = []

    for i in range(5):
        reconstructed_tensor = tensor_decomposition(X, W)

        # 论文第七页，Eq.12
        # v_bef, v_aft = reconstructed_tensor[0, 0, k], reconstructed_tensor[0, 1, k]
        # v_est = v_bef + v_aft
        if not np.isnan(sum((reconstructed_tensor[0, 0, k] + reconstructed_tensor[0, 1, k]) for k in range(cycle_num))):
            cal_list.append(
                abs(round(sum((reconstructed_tensor[0, 0, k] + reconstructed_tensor[0, 1, k]) for k in range(cycle_num)))))
        else:
            cal_list.append(np.nan)

    Q.append({
        'Link ID': link_id,
        'Lane ID': lane_id,
        'est_flow_1': cal_list[0],
        'est_flow_2': cal_list[1],
        'est_flow_3': cal_list[2],
        'est_flow_4': cal_list[3],
        'est_flow_5': cal_list[4]
    })

    # print(Q)
    Q_est = pd.DataFrame(Q)
    all_Q_est.append(Q_est)

final_Q_est = pd.concat(all_Q_est, ignore_index=True)
final_Q_est.to_csv(file_save , index=False)
# print(final_Q_est)
