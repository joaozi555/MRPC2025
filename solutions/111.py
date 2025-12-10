import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def quat_to_matrix(q):
    #四元数转旋转矩阵
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]
    ])


def matrix_to_quat(R):
    #旋转矩阵转四元数
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw])


def relative_rotation(t, omega, alpha):
    #计算末端执行器相对旋转矩阵
    cos_wt = np.cos(omega * t)
    sin_wt = np.sin(omega * t)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    return np.array([
        [cos_wt, -sin_wt * cos_a, sin_wt * sin_a],
        [sin_wt, cos_wt * cos_a, -cos_wt * sin_a],
        [0, sin_a, cos_a]
    ])


def main():
    #读取数据
    df = pd.read_csv('D:/MRPC-2025-homework-main/documents/tracking.csv')

    #参数设置
    OMEGA = (0.5) * np.pi  #角频率
    ALPHA = np.pi / 12  #固定角度

    results = []
    prev_q = None

    for _, row in df.iterrows():
        t = row['t']
        q_body = np.array([row['qx'], row['qy'], row['qz'], row['qw']])

        #本体四元数归一化
        q_body = q_body / np.linalg.norm(q_body)

        #转换到旋转矩阵
        R_WB = quat_to_matrix(q_body)

        #计算相对旋转
        R_BD = relative_rotation(t, OMEGA, ALPHA)

        #组合旋转
        R_WD = R_WB @ R_BD

        #转回四元数
        q_ee = matrix_to_quat(R_WD)

        #归一化
        q_ee = q_ee / np.linalg.norm(q_ee)

        #连续性处理：确保qw>=0
        if q_ee[3] < 0:
            q_ee = -q_ee

        results.append([t, *q_ee])

    #保存结果
    result_df = pd.DataFrame(results, columns=['t', 'qx', 'qy', 'qz', 'qw'])

    #绘图
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for i, comp in enumerate(['qx', 'qy', 'qz', 'qw']):
        axes[i].plot(result_df['t'], result_df[comp], 'b-', linewidth=1.5)
        axes[i].set_ylabel(f'${comp}$', fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', labelsize=10)

    axes[3].set_xlabel('Time $t$ (s)', fontsize=12)
    axes[0].set_title('End-Effector Attitude Quaternions', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('end_effector_quaternions.png', dpi=300, bbox_inches='tight')
    plt.show()

    return result_df

#执行
if __name__ == '__main__':
    df_result = main()