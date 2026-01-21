# Search-YOPO 2D 深度解析

## 📋 文档概述

这是一个将YOPO_2D从**点对点导航**改造为**主动搜索规划**的技术方案。核心思想是让机器人在**未知环境中自主搜索静态信号源**，而不是导航到已知目标点。

---

## 🎯 核心改造思路

### 从导航到搜索的范式转变

```
原版 YOPO_2D:
"我知道目标在哪，帮我规划一条安全的路径到达那里"
目标: [x_goal, y_goal] (已知)
引导: 欧氏距离最小化

↓ 改造为 ↓

Search-YOPO 2D:
"我不知道目标在哪，帮我规划一条路径去找到它"
目标: 信号源位置 (未知)
引导: 信息增益最大化
```

---

## 🏗️ 系统架构改造

### 1. 问题建模 (2D)

#### 1.1 任务定义

```
环境: Ω ⊂ ℝ² (2D平面)
机器人: 质点或圆形 (半径r)
目标: 静态信号源 (位置未知)
障碍物: 未知分布
任务: 在避障前提下，最快找到信号源
```

#### 1.2 新增传感器模型

除了原有的**激光雷达**，增加**信号传感器**:

| 属性 | 规格 | 说明 |
|------|------|------|
| 类型 | 全向接收机 | 360度接收信号 |
| 输出 | z_t ∈ {0,1} 或 ℝ | 二值检测或强度值 |
| 物理模型 | P_r ∝ 1/d²_target | 信号强度与距离平方成反比 |
| 作用 | 提供目标方向的概率信息 | 更新信念图 |

**信号强度模型**:

```python
# 路径损耗模型 (简化版)
P_r(d) = P_t × (d_0 / d)^n
# P_t: 发射功率
# d_0: 参考距离 (如1m)
# n: 路径损耗指数 (通常2-4)
# d: 距离信号源的距离

# 观测模型
z_t = {
    1 (检测到),  if P_r(d) > threshold
    0 (未检测),  otherwise
}
```

#### 1.3 信念图 (Belief Map)

机器人维护一个**2D概率栅格地图**，实时更新对目标位置的信念:

```python
# 信念图维护
M_prob: [W_map × H_map]  # 每个栅格的目标概率
M_ent:  [W_map × H_map]  # 每个栅格的信息熵

# 贝叶斯更新 (对数几率形式)
l_t(x,y) = l_{t-1}(x,y) + log(P(z_t|target at (x,y)) / P(z_t|no target))

# 概率归一化
P_t(x,y) = exp(l_t(x,y)) / Σ exp(l_t)
```

**熵图计算**:

```python
# Shannon熵: 衡量不确定性
H(x,y) = -P(x,y)×log(P(x,y)) - (1-P(x,y))×log(1-P(x,y))

# 未探索区域: H ≈ 1 (最大不确定性)
# 已确认区域: H ≈ 0 (高度确定)
```

---

## 🧠 网络架构改造

### 2.1 输入层改造

#### 原版输入
```python
Input_original = {
    'lidar': [B, 360],        # 激光雷达扫描
    'state': [B, N_prim, 6]   # [vx, vy, ax, ay, gx, gy]
}
```

#### 改造后输入
```python
Input_search = {
    'lidar': [B, 360],              # 激光雷达 (保持不变)
    'local_map': [B, 2, 64, 64],    # 局部信息图 (新增)
    'state': [B, N_prim, 4]         # [vx, vy, ax, ay] (移除目标点)
}

# 局部信息图详解
local_map[0] = P_local  # Channel 0: 局部概率图
local_map[1] = H_local  # Channel 1: 局部熵图

# 裁剪参数
map_size: 64×64 pixels
resolution: 0.2 m/pixel
coverage: 12.8m × 12.8m (以机器人为中心)
```

#### 局部地图裁剪流程

```python
def extract_local_map(global_map, robot_pos, robot_heading, map_size=64):
    """
    从全局地图裁剪出以机器人为中心、对齐机体系的局部地图
    
    Args:
        global_map: [H_global, W_global, 2] 全局信念图
        robot_pos: [x, y] 机器人世界坐标
        robot_heading: θ 机器人朝向
        map_size: 局部地图大小
    
    Returns:
        local_map: [2, map_size, map_size] 局部地图
    """
    # 1. 构建仿射变换矩阵
    # (平移到机器人位置 + 旋转到机体系)
    cos_theta = np.cos(-robot_heading)  # 逆旋转
    sin_theta = np.sin(-robot_heading)
    
    affine_matrix = [
        [cos_theta, -sin_theta, -robot_pos[0]],
        [sin_theta,  cos_theta, -robot_pos[1]]
    ]
    
    # 2. 使用双线性插值裁剪
    grid = F.affine_grid(affine_matrix, [1, 2, map_size, map_size])
    local_map = F.grid_sample(global_map, grid, mode='bilinear')
    
    return local_map
```

### 2.2 双流骨干网络 (Dual-Stream Backbone)

由于**激光雷达是1D序列**，**地图是2D图像**，采用双流架构:

```
┌────────────────────────────────────────────────────────┐
│              Search-YOPO Network                       │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌─────────────────┐      ┌──────────────────┐        │
│  │  Lidar Stream   │      │   Map Stream     │        │
│  │  [B, 360]       │      │  [B, 2, 64, 64]  │        │
│  └────────┬────────┘      └────────┬─────────┘        │
│           │                        │                  │
│           ▼                        ▼                  │
│  ┌─────────────────┐      ┌──────────────────┐        │
│  │  ResNet1D       │      │  Lightweight     │        │
│  │  360 → 128      │      │  CNN             │        │
│  │                 │      │  2×64×64 → 128   │        │
│  └────────┬────────┘      └────────┬─────────┘        │
│           │                        │                  │
│           │    f_lidar   f_map     │                  │
│           └───────┬────────┬───────┘                  │
│                   ▼        ▼                          │
│            ┌──────────────────┐                       │
│            │   Concatenate    │                       │
│            │   [128 + 128]    │                       │
│            └────────┬─────────┘                       │
│                     ▼                                 │
│            ┌──────────────────┐                       │
│            │   Fusion Layer   │                       │
│            │   256 → 128      │                       │
│            └────────┬─────────┘                       │
│                     │                                 │
│                     ├─────┬─────┬─────┬─────┬─────┐   │
│                     ▼     ▼     ▼     ▼     ▼     ▼   │
│            ┌────────────────────────────────────┐    │
│            │  7 × Shared Head (MLP)             │    │
│            │  Concat with State [128+4]         │    │
│            └────────┬───────────────────────────┘    │
│                     ▼                                 │
│            Output: [B, 7, 7]                          │
│            [δyaw, δr, vx, vy, ax, ay, score]          │
└────────────────────────────────────────────────────────┘
```

#### Map Stream 实现

```python
class LightweightCNN(nn.Module):
    """轻量级2D CNN for 局部地图特征提取"""
    def __init__(self, in_channels=2, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            # Layer 1: 64×64 → 32×32
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Layer 2: 32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 3: 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Global Average Pooling: 8×8 → 1×1
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.fc = nn.Linear(128, hidden_dim)
    
    def forward(self, x):
        # x: [B, 2, 64, 64]
        features = self.encoder(x)  # [B, 128]
        return self.fc(features)     # [B, hidden_dim]
```

#### 融合策略

```python
class SearchYOPONetwork(nn.Module):
    def __init__(self):
        self.lidar_backbone = ResNet1D(...)
        self.map_backbone = LightweightCNN(...)
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.heads = nn.ModuleList([YopoHead2D(...) for _ in range(7)])
    
    def forward(self, lidar, local_map, state):
        # 双流特征提取
        f_lidar = self.lidar_backbone(lidar)  # [B, 128]
        f_map = self.map_backbone(local_map)   # [B, 128]
        
        # 特征融合
        f_fused = self.fusion(torch.cat([f_lidar, f_map], dim=-1))  # [B, 128]
        
        # 多头输出
        outputs = []
        for i, head in enumerate(self.heads):
            # 拼接当前基元的状态
            state_i = state[:, i, :]  # [B, 4]
            head_input = torch.cat([f_fused, state_i], dim=-1)  # [B, 132]
            output_i = head(head_input)  # [B, 7]
            outputs.append(output_i)
        
        outputs = torch.stack(outputs, dim=1)  # [B, 7, 7]
        
        # 分离偏移和评分
        offsets = outputs[:, :, :6]  # [B, 7, 6]
        scores = outputs[:, :, 6]     # [B, 7]
        
        return offsets, scores
```

### 2.3 输出层 (保持不变)

```python
Output = {
    'offsets': [B, N_prim, 6],  # [δyaw, δr, vx, vy, ax, ay]
    'scores': [B, N_prim]        # 轨迹评分 (信息增益-代价比)
}
```

**关键差异**: 
- 原版: `score` 代表到达目标的代价 (越小越好)
- 搜索版: `score` 代表信息增益与代价的权衡 (越小代价越低+信息越多)

---

## 🎯 核心创新: 信息势场引导

### 3.1 信息势场定义

这是**最核心的改造**，替代原有的"目标点欧氏距离"引导。

#### 势场公式

```python
# 信息势场 (值越小越好，负的信息增益)
V_info(x, y) = -(α·P_GT(x,y) + β·H_GT(x,y))

参数:
- P_GT(x,y): Ground Truth目标概率分布 (训练时特权信息)
- H_GT(x,y): Ground Truth熵图 (全局不确定性分布)
- α: 目标概率权重 (如 1.0)
- β: 探索权重 (如 0.5)
```

#### 物理意义

```
V_info < 0 的区域:
- 高目标概率 OR 高不确定性
- 机器人应该朝这些区域移动

V_info ≈ 0 的区域:
- 低目标概率 AND 已探索
- 机器人应该避开

梯度方向 -∇V_info:
- 指向信息增益最大的方向
- 自然形成"信息势场"吸引力
```

#### Ground Truth 生成 (训练时)

```python
def generate_GT_potential_field(target_pos, explored_region, map_size):
    """
    生成Ground Truth信息势场
    
    Args:
        target_pos: [x_t, y_t] 真实目标位置
        explored_region: [H, W] 已探索区域掩码
        map_size: (H, W)
    
    Returns:
        V_GT: [H, W] Ground Truth势场
    """
    H, W = map_size
    
    # 1. 生成目标概率图 (高斯分布)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    P_GT = np.exp(-((xx - target_pos[0])**2 + 
                    (yy - target_pos[1])**2) / (2 * sigma**2))
    P_GT = P_GT / P_GT.sum()  # 归一化
    
    # 2. 生成熵图
    H_GT = np.ones_like(P_GT)  # 初始全1 (完全不确定)
    H_GT[explored_region] = 0   # 已探索区域熵为0
    
    # 3. 组合势场
    V_GT = -(alpha * P_GT + beta * H_GT)
    
    return V_GT, P_GT, H_GT
```

### 3.2 搜索损失函数 L_search

将轨迹在信息势场中的"势能积分"作为损失:

#### 数学形式

```python
# 连续形式 (理论)
L_search = ∫₀ᵀ V_info(x(t), y(t)) dt

# 离散形式 (实现)
L_search ≈ (1/N_eval) × Σᵢ V_info(pᵢ)

其中:
- p(t) = [x(t), y(t)]ᵀ : 五次多项式轨迹
- pᵢ: 轨迹上的采样点 (如N_eval=20)
- T: 轨迹时间段
```

#### 代码实现

```python
def compute_search_loss(trajectory_points, V_GT, resolution):
    """
    计算搜索损失
    
    Args:
        trajectory_points: [N_eval, 2] 轨迹采样点 (世界坐标)
        V_GT: [H, W] Ground Truth势场
        resolution: 地图分辨率 (m/pixel)
    
    Returns:
        L_search: scalar 搜索损失
    """
    # 1. 世界坐标 → 栅格坐标
    grid_coords = trajectory_points / resolution
    
    # 2. 双线性插值采样势场值
    V_sampled = bilinear_interpolate(V_GT, grid_coords)  # [N_eval]
    
    # 3. 平均
    L_search = V_sampled.mean()
    
    return L_search

def bilinear_interpolate(image, coords):
    """双线性插值 (支持梯度反传)"""
    # coords: [N, 2] (x, y)
    x, y = coords[:, 0], coords[:, 1]
    
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    
    # 边界检查
    x0 = torch.clamp(x0, 0, image.shape[1]-1)
    x1 = torch.clamp(x1, 0, image.shape[1]-1)
    y0 = torch.clamp(y0, 0, image.shape[0]-1)
    y1 = torch.clamp(y1, 0, image.shape[0]-1)
    
    # 双线性插值
    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]
    
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())
    
    return wa*Ia + wb*Ib + wc*Ic + wd*Id
```

### 3.3 梯度反向传播

#### 链式法则推导

```
目标: 计算 ∂L_search/∂d
(d 是网络输出的边界条件 [p₀, v₀, a₀, p₁, v₁, a₁])

链式法则:
∂L_search/∂d = Σᵢ (∂L_search/∂pᵢ × ∂pᵢ/∂d)

其中:
1. ∂L_search/∂pᵢ = (1/N_eval) × ∇V_info(pᵢ)
   → 势场的空间梯度 [∂V/∂x, ∂V/∂y]ᵀ

2. ∂pᵢ/∂d = Jacobian of polynomial
   → 五次多项式对边界条件的雅可比矩阵
```

#### 实现方式

**方式1: 自动微分 (推荐)**

```python
# PyTorch自动处理梯度
trajectory_points = polynomial.evaluate(t_samples)  # 可微
V_sampled = F.grid_sample(V_GT, trajectory_points)  # 可微
L_search = V_sampled.mean()  # 可微
L_search.backward()  # 自动计算所有梯度!
```

**方式2: 手动计算梯度 (更高效)**

```python
def compute_search_loss_with_gradient(trajectory, V_GT, grad_V_GT):
    """
    手动计算搜索损失及梯度
    
    Args:
        trajectory: Poly5Trajectory object
        V_GT: [H, W] 势场
        grad_V_GT: [2, H, W] 势场梯度 [∂V/∂x, ∂V/∂y]
    
    Returns:
        L_search: scalar
        grad_d: [12] 对边界条件的梯度
    """
    # 1. 采样轨迹
    t_samples = np.linspace(0, T, N_eval)
    points = trajectory.evaluate(t_samples)  # [N_eval, 2]
    
    # 2. 采样势场值
    V_vals = sample_field(V_GT, points)  # [N_eval]
    L_search = V_vals.mean()
    
    # 3. 采样势场梯度
    grad_V_x = sample_field(grad_V_GT[0], points)  # [N_eval]
    grad_V_y = sample_field(grad_V_GT[1], points)  # [N_eval]
    
    # 4. 计算雅可比矩阵 ∂p/∂d
    jacobians = trajectory.compute_jacobian(t_samples)  # [N_eval, 2, 12]
    
    # 5. 链式法则
    grad_d = np.zeros(12)
    for i in range(N_eval):
        grad_V = np.array([grad_V_x[i], grad_V_y[i]])  # [2]
        grad_d += (1/N_eval) * (grad_V @ jacobians[i])  # [12]
    
    return L_search, grad_d
```

#### 势场梯度预计算

```python
def compute_potential_gradient(V_GT):
    """使用Sobel算子计算势场梯度"""
    grad_x = cv2.Sobel(V_GT, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(V_GT, cv2.CV_64F, 0, 1, ksize=3)
    return np.stack([grad_x, grad_y], axis=0)  # [2, H, W]
```

### 3.4 总损失函数

替换原有的L_guidance:

```python
# 原版 YOPO_2D
L_total = w_s×L_smooth + w_a×L_acc + w_c×L_safety + w_g×L_guidance
#                                                          ↓
#                                                   (目标点距离)

# Search-YOPO 2D
L_total = w_s×L_smooth + w_a×L_acc + w_c×L_safety + w_search×L_search
#                                                          ↓
#                                                   (信息势场积分)

# 默认权重
w_s = 10.0      # 平滑性
w_a = 1.0       # 加速度
w_c = 1.0       # 安全性
w_search = 0.5  # 搜索引导 (相比原w_g=0.15更大,因为搜索更重要)
```

---

## 📦 数据集生成改造

### 4.1 从静态到序列化

#### 原版数据集 (静态单帧)

```python
# 原版: 独立采样
for i in range(N_samples):
    pos = sample_random_position()
    lidar = get_lidar_scan(pos)
    goal = sample_random_goal()
    dataset.append((lidar, pos, heading, vel, acc, goal))
```

#### 搜索版数据集 (序列化)

```python
# 搜索版: 序列化仿真
for episode in range(N_episodes):
    # 初始化
    map = generate_forest_map()
    target_pos = place_random_target()
    robot_pos = random_start_position()
    belief_map = np.ones((H, W)) * 0.5  # 均匀先验
    
    trajectory = []
    
    # 专家策略运行 (如Frontier-based或Info-taxis)
    while not found_target:
        # 1. 传感器观测
        lidar_scan = simulate_lidar(robot_pos, map)
        signal_reading = simulate_signal(robot_pos, target_pos)
        
        # 2. 信念更新
        belief_map = bayesian_update(belief_map, signal_reading, robot_pos)
        entropy_map = compute_entropy(belief_map)
        
        # 3. 裁剪局部地图
        local_map = extract_local_map(
            np.stack([belief_map, entropy_map]),
            robot_pos, robot_heading
        )
        
        # 4. 专家决策
        action = expert_policy(lidar_scan, local_map, belief_map)
        
        # 5. 执行并记录
        robot_pos, robot_vel, robot_acc = execute_action(action)
        
        # 6. 生成GT势场 (用于训练)
        V_GT, P_GT, H_GT = generate_GT_potential_field(
            target_pos, belief_map
        )
        
        # 7. 保存数据
        trajectory.append({
            'lidar': lidar_scan,
            'local_map': local_map,
            'robot_state': [robot_pos, robot_heading, robot_vel, robot_acc],
            'V_GT': V_GT,
            'P_GT': P_GT,
            'H_GT': H_GT,
            'map': map  # 用于ESDF计算
        })
    
    dataset.append(trajectory)
```

### 4.2 专家策略选择

#### 选项1: Frontier-based探索

```python
def frontier_based_expert(lidar, belief_map):
    """
    基于前沿的探索策略
    - 寻找已探索与未探索的边界
    - 选择最近的前沿点作为目标
    """
    frontiers = detect_frontiers(belief_map)
    nearest_frontier = find_nearest(frontiers, robot_pos)
    return plan_to_goal(nearest_frontier)
```

#### 选项2: Information-driven (Info-taxis)

```python
def info_taxis_expert(belief_map, entropy_map):
    """
    信息驱动的策略
    - 计算每个候选动作的期望信息增益
    - 选择信息增益最大的动作
    """
    actions = generate_candidate_actions()
    info_gains = [compute_expected_info_gain(a) for a in actions]
    best_action = actions[np.argmax(info_gains)]
    return best_action
```

### 4.3 数据增强策略

```python
# 离线训练时可以打散序列
def create_training_batches(episodes):
    """
    将序列化数据打散为独立样本 (Off-policy)
    """
    all_frames = []
    for episode in episodes:
        for frame in episode:
            all_frames.append(frame)
    
    # 随机打乱
    random.shuffle(all_frames)
    
    # 构建batches
    return DataLoader(all_frames, batch_size=64, shuffle=True)
```

---

## 🎓 训练流程

### 5.1 训练循环

```python
def train_search_yopo(model, dataloader, optimizer, device):
    for epoch in range(num_epochs):
        for batch in dataloader:
            # 1. 加载数据
            lidar = batch['lidar'].to(device)          # [B, 360]
            local_map = batch['local_map'].to(device)  # [B, 2, 64, 64]
            robot_state = batch['robot_state']
            V_GT = batch['V_GT'].to(device)            # [B, H, W]
            map_data = batch['map']
            
            # 2. 构建状态输入
            # 注意: 不再有goal,只有速度和加速度
            state_body = build_state(robot_state)  # [B, 7, 4]
            
            # 3. 网络前向
            offsets, scores = model(lidar, local_map, state_body)
            
            # 4. 生成候选轨迹
            trajectories = generate_trajectories(
                robot_state, offsets, T_segment
            )  # [B, 7, N_eval, 2]
            
            # 5. 计算多目标损失
            losses = {}
            
            # 5.1 搜索损失
            for i in range(7):  # 对每个基元
                traj_points = trajectories[:, i, :, :]  # [B, N_eval, 2]
                L_search_i = compute_search_loss(traj_points, V_GT)
                losses[f'search_{i}'] = L_search_i
            
            # 5.2 安全损失
            for i in range(7):
                L_safety_i = compute_safety_loss(
                    trajectories[:, i], map_data
                )
                losses[f'safety_{i}'] = L_safety_i
            
            # 5.3 平滑损失
            for i in range(7):
                L_smooth_i = compute_smoothness_loss(
                    trajectories[:, i]
                )
                losses[f'smooth_{i}'] = L_smooth_i
            
            # 5.4 加速度损失
            for i in range(7):
                L_acc_i = compute_acceleration_loss(
                    trajectories[:, i]
                )
                losses[f'acc_{i}'] = L_acc_i
            
            # 6. 总损失 (所有基元求和)
            L_total_per_prim = torch.stack([
                w_search * losses[f'search_{i}'] +
                w_safety * losses[f'safety_{i}'] +
                w_smooth * losses[f'smooth_{i}'] +
                w_acc * losses[f'acc_{i}']
                for i in range(7)
            ])  # [7]
            
            L_trajectory = L_total_per_prim.mean()
            
            # 7. Score监督损失
            # 目标: 让网络预测的score接近真实总损失
            score_target = L_total_per_prim.detach()  # [7]
            L_score = F.smooth_l1_loss(scores, score_target)
            
            # 8. 最终损失
            L_total = L_trajectory + L_score
            
            # 9. 反向传播
            optimizer.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
            # 10. 日志记录
            if step % 100 == 0:
                print(f'Epoch {epoch}, Step {step}:')
                print(f'  L_total: {L_total.item():.4f}')
                print(f'  L_search: {L_trajectory.item():.4f}')
                print(f'  L_score: {L_score.item():.4f}')
```

### 5.2 Score标签生成

```python
# 方式1: 直接使用总损失
score_target = L_total_per_primitive  # 越小越好

# 方式2: 负指数变换 (原文档建议)
score_target = torch.exp(-L_total_per_primitive)  # 越大越好
# 此时训练时应最大化score

# 推荐: 方式1更直观
```

---

## ⚙️ 配置文件修改

### config.yaml 新增部分

```yaml
# ==================== 搜索任务配置 ====================
task:
  mode: "search"  # "navigation" or "search"

# 信号传感器
signal_sensor:
  enabled: true
  type: "binary"           # "binary" or "continuous"
  detection_range: 20.0    # m
  noise_std: 0.1
  
  # 信号强度模型
  path_loss_exponent: 2.5
  reference_distance: 1.0  # m
  transmit_power: 1.0

# 信念图
belief_map:
  resolution: 0.2          # m/pixel
  initial_prob: 0.5        # 均匀先验
  update_method: "bayesian"  # "bayesian" or "particle_filter"

# 局部地图
local_map:
  size: 64                 # 64×64 pixels
  resolution: 0.2          # m/pixel → 覆盖12.8m×12.8m
  channels: 2              # [probability, entropy]

# ==================== 网络架构 ====================
network:
  input_type: "multimodal"  # "lidar_only" or "multimodal"
  
  # Lidar Stream (保持不变)
  lidar_backbone: "resnet1d"  # "resnet1d", "conv1d", "mlp"
  lidar_dim: 360
  lidar_hidden_dim: 128
  
  # Map Stream (新增)
  map_backbone: "lightweight_cnn"
  map_channels: 2
  map_size: 64
  map_hidden_dim: 128
  
  # Fusion
  fusion_dim: 128
  
  # State dimension
  state_dim: 4  # [vx, vy, ax, ay] (移除goal)

# ==================== 损失权重 ====================
loss_weights:
  search: 0.5        # 替代原w_guidance
  smoothness: 10.0
  acceleration: 1.0
  safety: 1.0
  score: 1.0         # score监督权重

# 信息势场
potential_field:
  alpha: 1.0         # 目标概率权重
  beta: 0.5          # 熵权重 (探索权重)

# ==================== 训练配置 ====================
training:
  dataset_type: "sequential"  # "static" or "sequential"
  expert_policy: "info_taxis"  # "frontier" or "info_taxis"
  
  # 序列数据
  num_episodes: 1000
  max_episode_steps: 500
  
  # 训练参数 (保持不变)
  batch_size: 64
  learning_rate: 1.5e-4
  epochs: 100

# ==================== 测试配置 ====================
testing:
  success_threshold: 2.0  # 距离目标<2m算成功
  max_steps: 1000
  visualize_belief: true  # 可视化信念图
```

---

## 📊 架构对比总结

### 完整对比表

| **特性** | **原版 YOPO_2D** | **Search-YOPO 2D** |
|---------|------------------|-------------------|
| **任务类型** | 点对点导航 | 主动搜索 |
| **目标信息** | 已知精确位置 | 未知,需搜索 |
| **输入数据** | Lidar (1D) + Goal向量 | Lidar (1D) + Local Map (2D) |
| **网络架构** | 单流ResNet1D/MLP | 双流(ResNet1D + CNN) |
| **状态维度** | 6 (v, a, goal) | 4 (v, a) |
| **引导机制** | 欧氏距离最小化 | 信息势场积分最小化 |
| **损失函数** | L_guidance (距离) | L_search (信息增益) |
| **训练数据** | 静态单帧采样 | 序列化仿真轨迹 |
| **专家策略** | 不需要 | Frontier/Info-taxis |
| **计算复杂度** | 低 (~5ms) | 中等 (~10ms) |
| **参数量** | ~2M | ~3M (增加CNN) |
| **适用场景** | 室内导航、路径规划 | 搜救、探索、信号追踪 |

### 核心优势对比

#### 原版YOPO_2D
✅ 计算速度极快  
✅ 实现简单  
✅ 适合已知目标导航  
❌ 无法处理未知目标  
❌ 无探索能力  

#### Search-YOPO 2D
✅ 主动探索未知环境  
✅ 信息驱动的规划  
✅ 保持YOPO单次推理优势  
✅ 适合搜救、探测任务  
⚠️ 需要更多训练数据  
⚠️ 计算略慢 (仍可实时)  

---

## 🔬 理论深度解析

### 为什么用信息势场而不是直接最大化信息增益?

#### 问题背景

传统信息规划方法:
```
动作选择 = argmax 期望信息增益(a)
问题: 需要在线遍历所有候选动作,计算慢
```

YOPO方式:
```
将"信息增益最大化"转化为"势场积分最小化"
优势: 单次神经网络前向推理,端到端可微
```

#### 数学等价性

```
原始目标: 最大化信息增益
max I(trajectory) = max ΔH(target|trajectory)

转化为: 最小化负信息势场
min L_search = min ∫ V_info dt
            = min ∫ -(α·P + β·H) dt
            = max ∫ (α·P + β·H) dt

其中:
- α·P: 朝向高目标概率区域 (Exploitation)
- β·H: 朝向高不确定性区域 (Exploration)
```

### 为什么需要Ground Truth势场?

#### 监督学习的本质

```
问题: 如何让网络学会"朝信息最多的方向规划"?

方案1 (强化学习):
- 用信息增益作为reward
- 缺点: 样本效率低,训练不稳定

方案2 (模仿学习 + GT势场):
- 训练时给网络"上帝视角"
- 告诉它"目标真的在这里,朝这个方向规划信息最多"
- 优点: 收敛快,稳定
- 测试时网络依靠learned features泛化
```

#### GT vs 实际信念图

```
训练时:
GT势场 = -(α·P_true + β·H_global)
↓ 网络学习
学到: "高P区域+高H区域 = 应该去的地方"

测试时:
输入: Local belief map (机器人自己维护)
输出: 基于learned features的规划
不需要GT,依靠泛化
```

---

## 🚀 实现建议与注意事项

### 1. 从简单到复杂的实现路线

#### Phase 1: 基础双流网络 (1-2周)

```python
# 目标: 跑通双流架构
1. 实现LightweightCNN
2. 修改YopoNetwork加入map stream
3. 用随机数据测试前向传播
4. 检查参数量和推理时间
```

#### Phase 2: 信念图维护 (1周)

```python
# 目标: 实现信念更新逻辑
1. 实现信号传感器仿真
2. 实现贝叶斯更新
3. 实现熵计算
4. 实现局部地图裁剪 (affine_grid)
```

#### Phase 3: 数据集生成 (2周)

```python
# 目标: 生成序列化训练数据
1. 实现简单的Frontier专家策略
2. 生成100条episode
3. 可视化验证数据质量
4. 检查信念图更新是否合理
```

#### Phase 4: 损失函数 (1周)

```python
# 目标: 实现L_search
1. 实现GT势场生成
2. 实现轨迹采样
3. 实现势场积分计算
4. 验证梯度反传
```

#### Phase 5: 完整训练 (2-3周)

```python
# 目标: 端到端训练
1. 整合所有模块
2. 调整超参数
3. 训练并观察收敛
4. TensorBoard可视化
```

### 2. 调试技巧

#### 可视化信念图更新

```python
import matplotlib.pyplot as plt

def visualize_belief_update(belief_before, belief_after, signal_reading):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(belief_before)
    axes[0].set_title('Before Update')
    
    axes[1].imshow(belief_after)
    axes[1].set_title(f'After Update (signal={signal_reading})')
    
    axes[2].imshow(belief_after - belief_before, cmap='RdBu')
    axes[2].set_title('Difference')
    
    plt.show()
```

#### 可视化势场和轨迹

```python
def visualize_potential_and_trajectory(V_GT, trajectory):
    plt.figure(figsize=(10, 10))
    
    # 势场热力图
    plt.imshow(V_GT, cmap='viridis', alpha=0.7)
    plt.colorbar(label='V_info')
    
    # 轨迹
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2)
    plt.scatter(trajectory[0, 0], trajectory[0, 1], c='g', s=100, label='Start')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='r', s=100, label='End')
    
    plt.legend()
    plt.title('Potential Field + Trajectory')
    plt.show()
```

### 3. 超参数调优建议

| 参数 | 初始值 | 调优范围 | 影响 |
|------|--------|---------|------|
| α (目标权重) | 1.0 | [0.5, 2.0] | 太大→只追目标,不探索 |
| β (熵权重) | 0.5 | [0.1, 1.0] | 太大→盲目探索 |
| w_search | 0.5 | [0.3, 1.0] | 搜索vs安全的权衡 |
| map_size | 64 | [32, 128] | 感受野大小 |
| resolution | 0.2 | [0.1, 0.5] | 精度vs效率 |

### 4. 常见问题

#### Q1: 信念图不收敛?
```python
# 检查:
1. 信号模型是否合理 (P_r vs distance)
2. 贝叶斯更新是否正确 (对数几率形式)
3. 归一化是否丢失概率质量
```

#### Q2: 网络输出全0或全NaN?
```python
# 检查:
1. 局部地图归一化 ([0,1]范围)
2. 损失scale是否合理 (不同损失项数量级)
3. 梯度裁剪是否过小
```

#### Q3: 训练loss不下降?
```python
# 尝试:
1. 降低学习率到1e-5
2. 增加w_search权重
3. 检查专家策略质量
4. 增加训练数据量
```

---

## 📖 参考文献与扩展阅读

### 核心论文

1. **YOPO原文**: "You Only Plan Once: ..."
2. **信息驱动规划**: "Information-Theoretic Planning with Trajectory Optimization"
3. **Frontier探索**: "Frontier-Based Exploration Using Multiple Robots"
4. **信念空间规划**: "Belief Space Planning"

### 相关算法

| 算法 | 特点 | 与本方案关系 |
|------|------|-------------|
| Frontier-based | 简单高效 | 可作为专家策略 |
| Info-taxis | 理论最优 | 本方案的监督信号来源 |
| SLAM | 地图构建 | 可结合使用 |
| A* | 全局规划 | 可作为baseline对比 |

---

## 🎓 总结

Search-YOPO 2D通过以下关键改造,将YOPO从导航扩展到搜索:

### 核心创新
1. **双流网络**: Lidar + Local Map融合
2. **信息势场**: 替代欧氏距离的引导机制
3. **序列化数据**: 从静态采样到动态探索轨迹
4. **端到端可微**: 保持YOPO的速度优势

### 理论意义
- 将信息理论与深度学习结合
- 实现"单次规划"的探索任务
- 提供新的搜索问题建模范式

### 实际价值
- 搜救机器人快速部署
- 信号源追踪 (辐射、化学泄漏)
- 环境监测和探索

**这个方案完美融合了YOPO的速度优势和信息规划的智能,是一个非常有价值的研究方向!** 🚀
