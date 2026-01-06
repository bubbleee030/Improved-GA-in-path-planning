import numpy as np
import matplotlib.pyplot as plt

# 論文中的核心改進公式 (Equations 10 & 11)

def improved_crossover_rate(i, pd=0.8):
    """
    論文公式 (10): 非線性交配率
    i: 當前迭代次數
    pd: 預設交配率閾值 (本範例設為 0.8)
    """
    # 這裡忽略了 epsilon 誤差項，專注於主要趨勢
    term = np.sqrt(i) / ((1 + i)**2)
    p_c = -term + pd
    return p_c

def improved_mutation_rate(i, G, p_min=0.01):
    """
    論文公式 (11): 線性(或非線性)變異率
    i: 當前迭代次數
    G: 最大迭代次數
    p_min: 最小變異率
    """
    p_m = p_min * (1 - np.sqrt(i / G))
    return p_m

# 路徑平滑度計算 (Fitness Function Part)

def calculate_smoothness_penalty(path_coords):
    """
    模擬論文中的平滑度懲罰函數 (Eq 8 & 9)
    path_coords: 路徑點座標列表 [(x1,y1), (x2,y2), ...]
    """
    penalty = 0
    # 遍歷路徑中的每一個轉折點 (除去起點和終點)
    for i in range(1, len(path_coords) - 1):
        p_prev = np.array(path_coords[i-1])
        p_curr = np.array(path_coords[i])
        p_next = np.array(path_coords[i+1])
        
        # 計算向量
        vec1 = p_curr - p_prev
        vec2 = p_next - p_curr
        
        # 計算轉向角度 theta (利用向量夾角公式)
        # cos_theta = (a . b) / (|a| * |b|)
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        if norm_a == 0 or norm_b == 0:
            continue
            
        cos_theta = dot_product / (norm_a * norm_b)
        # 限制範圍避免數值誤差
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # 算出角度 (弧度轉角度)
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        
        # 論文邏輯：根據角度給予不同懲罰 (模擬論文中的分段函數) VFH 方法
        # 角度越小(急轉彎)，懲罰越大
        # 註：這裡僅為示意，論文有具體的 alpha, beta, gamma 值
        turn_angle = 180 - angle_deg # 轉彎幅度
        
        if turn_angle > 90: # 急轉彎 (>90度)
            penalty += 100 
        elif turn_angle > 45: # 中等轉彎
            penalty += 50
        else: # 平滑轉彎
            penalty += 10
            
    return penalty

# Demo and Visualization

# 設定參數
Max_Generations = 50
iterations = np.arange(1, Max_Generations + 1)

# 計算每一代的參數變化
pc_values = [improved_crossover_rate(i) for i in iterations]
pm_values = [improved_mutation_rate(i, Max_Generations) for i in iterations]

# 繪圖
plt.figure(figsize=(12, 5))

# 圖 1: 交配率變化
plt.subplot(1, 2, 1)
plt.plot(iterations, pc_values, 'b-', linewidth=2)
plt.title('Improved Crossover Rate ($p_c$)', fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.grid(True)

# 圖 2: 變異率變化
plt.subplot(1, 2, 2)
plt.plot(iterations, pm_values, 'r-', linewidth=2)
plt.title('Improved Mutation Rate ($p_m$)', fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.grid(True)

plt.tight_layout()
plt.show()

# 測試平滑度計算
sample_path = [(0,0), (1,1), (1,2), (2,2), (2,1)] # 一條包含直角和回頭的折線
score = calculate_smoothness_penalty(sample_path)
print(f"Sample Path: {sample_path}")
print(f"Smoothness Penalty Score: {score}")