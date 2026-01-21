# YOPO 2D 项目详细理论文档

## 1. 项目概述

**YOPO (You Only Plan Once)** 是一种基于深度学习的端到端运动规划方法。YOPO_2D 是原始 3D 版本 (YOPO_Sim) 的 2D 简化实现，采用纯 Python 编写，无需 ROS 依赖，适合快速原型验证和算法学习。

### 1.1 核心思想

YOPO 的核心理念是**单次前向推理完成轨迹规划**，与传统的迭代优化方法（如 MPC）相比，具有更低的计算延迟。其主要特点包括：

1. **运动基元表示**：使用预定义的极坐标基元网格覆盖规划空间
2. **神经网络回归**：网络直接输出各基元的终止状态偏移和评分
3. **五次多项式轨迹**：使用闭式求解生成平滑的五次多项式轨迹
4. **多目标损失**：结合安全性、平滑性、引导性的综合损失函数

### 1.2 项目结构

```
YOPO_2D/
├── config/
│   ├── __init__.py         # 配置管理器
│   └── config.yaml         # 核心配置文件
├── simulator/
│   ├── map_generator.py    # 2D地图生成器 (森林、迷宫等)
│   ├── sensor.py           # 2D激光雷达仿真
│   └── dynamics.py         # 机器人动力学 + 多项式求解器
├── policy/
│   ├── primitive.py        # 2D运动基元定义
│   ├── network.py          # YOPO神经网络架构
│   └── loss.py             # 多目标损失函数
├── dataset.py              # 数据集生成和加载
├── train.py                # 训练脚本
├── test.py                 # 测试和可视化
└── README.md
```

---

## 2. 运动基元理论

### 2.1 基元定义

运动基元采用**极坐标表示**，每个基元由一个锚点角度 $\alpha_i$ 定义，锚点位于以机器人为中心、半径为 $r_{radio}$ 的圆弧上。

**参数定义：**

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 基元数量 | $N_{horizon}$ | 7 | 水平方向基元数量 |
| 规划半径 | $r_{radio}$ | 5.0 m | 基元锚点到机器人的距离 |
| 总覆盖角度 | $\theta_{fov}$ | 140° | 所有基元覆盖的总角度范围 |
| 锚点FOV | $\theta_{anchor}$ | 30° | 单个基元的角度变化范围 |

**锚点角度计算：**

对于第 $j$ 个基元 ($j = 0, 1, ..., N_{horizon}-1$)，其锚点角度为：

$$\alpha_j = -\frac{\theta_{fov}}{2} + j \cdot \frac{\theta_{fov}}{N_{horizon}}$$

其中 $j=0$ 对应最右侧（负角度），$j=N_{horizon}-1$ 对应最左侧（正角度）。

**锚点位置（机体坐标系）：**

$$\mathbf{p}_j^{anchor} = r_{radio} \cdot [\cos(\alpha_j), \sin(\alpha_j)]^T$$

### 2.2 网络输出与终止状态

网络对每个基元输出 7 维向量（经过 tanh 激活）：

$$\mathbf{o}_j = [\delta_{yaw}, \delta_r, v_x, v_y, a_x, a_y, s]$$

其中：
- $\delta_{yaw} \in [-1, 1]$：角度偏移（归一化）
- $\delta_r \in [-1, 1]$：径向偏移（归一化）
- $(v_x, v_y, a_x, a_y) \in [-1, 1]^4$：终止速度和加速度（归一化）
- $s$：轨迹评分（softplus 激活）

**反归一化过程：**

1. **终止位置**：
   $$\psi_j = \alpha_j + \delta_{yaw} \cdot \Delta\psi_{max}$$
   $$r_j = (\delta_r + 1) \cdot r_{radio}$$
   $$\mathbf{p}_j^{end} = [r_j \cos(\psi_j), r_j \sin(\psi_j)]^T$$

   其中 $\Delta\psi_{max} = \frac{\theta_{anchor}}{2}$ 是单个基元的最大角度偏移。

2. **终止速度和加速度**：
   $$\mathbf{v}_j^{end} = [v_x \cdot v_{max}, v_y \cdot v_{max}]^T$$
   $$\mathbf{a}_j^{end} = [a_x \cdot a_{max}, a_y \cdot a_{max}]^T$$

### 2.3 坐标系变换

YOPO 使用两套坐标系：

1. **机体坐标系 (Body Frame)**：以机器人中心为原点，前进方向为 X 轴
2. **基元坐标系 (Primitive Frame)**：以机器人为原点，各基元的锚点方向为 X 轴

**状态变换流程：**

```
世界系状态 → 机体系状态 → 各基元系状态 → 网络输入
                                ↓
                           网络输出
                                ↓
基元系终止状态 → 机体系终止状态 → 世界系终止状态
```

**旋转矩阵**：

从机体系到基元系（绕原点逆时针旋转 $-\alpha_j$）：

$$R_{body \to prim}^j = \begin{bmatrix} \cos(-\alpha_j) & -\sin(-\alpha_j) \\ \sin(-\alpha_j) & \cos(-\alpha_j) \end{bmatrix}$$

---

## 3. 神经网络架构

### 3.1 网络总体结构

```
┌─────────────────────────────────────────────────────────────┐
│                      YopoNetwork2D                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Lidar Input [B, 360]                                      │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────────────┐                                   │
│   │  Lidar Backbone     │  (ResNet1D/Conv1D/MLP)            │
│   │  360 → 128          │                                   │
│   └─────────────────────┘                                   │
│         │                                                   │
│         ▼                                                   │
│   Lidar Features [B, 128]                                   │
│         │                                                   │
│         ├──────────────┬──────────────┬─────────...         │
│         ▼              ▼              ▼                     │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│   │ Concat    │  │ Concat    │  │ Concat    │   × N_prim    │
│   │ + Head    │  │ + Head    │  │ + Head    │               │
│   └───────────┘  └───────────┘  └───────────┘               │
│         │              │              │                     │
│         ▼              ▼              ▼                     │
│   [B, 7]         [B, 7]         [B, 7]                      │
│                                                             │
│   State Input [B, N_prim, 6]                                │
│         │                                                   │
│         ▼ (per-primitive concatenation)                     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│   Output: endstate [B, N, 6] + score [B, N]                 │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 骨干网络 (Backbone)

支持三种骨干网络架构：

#### 3.2.1 ResNet1D（默认）

```python
ResNet1DBackbone:
├── Stem: Conv1d(1, 64, k=7, s=2) + BN + ReLU
├── Layer1: 2 × BasicBlock1D(64, 64)
├── Layer2: 2 × BasicBlock1D(64, 128, stride=2)
├── Layer3: 2 × BasicBlock1D(128, 256, stride=2)
├── Layer4: 2 × BasicBlock1D(256, 512)
├── AdaptiveAvgPool1d(1)
└── Linear(512, hidden_dim)
```

**BasicBlock1D** 结构：
$$y = \text{ReLU}(\text{BN}(\text{Conv}(x)) + \text{Shortcut}(x))$$

#### 3.2.2 Conv1D

```python
Conv1DBackbone:
├── Stem: Conv1d(1→32, k=7, s=2) + Conv1d(32→64, k=5, s=2)
├── Stage1: 2 × ResidualConv1DBlock(64)
├── Down1: Conv1d(64→128, s=2)
├── Stage2: 2 × ResidualConv1DBlock(128)
├── Down2: Conv1d(128→256, s=2)
├── Stage3: 1 × ResidualConv1DBlock(256)
├── AdaptiveAvgPool1d(1)
└── Linear(256, hidden_dim)
```

#### 3.2.3 MLP

```python
MLPBackbone:
├── Linear(360, 256) + ReLU
├── 3 × ResidualMLPBlock(256)
└── Linear(256, hidden_dim)
```

### 3.3 输出头 (Head)

每个基元共享同一个输出头网络：

```python
YopoHead2D:
├── Linear(hidden_dim + 6, 256) + ReLU
├── Linear(256, 256) + ReLU
├── Linear(256, 128) + ReLU
└── Linear(128, 7)
```

输出分解：
- 前 6 维：$\tanh(\cdot)$ 激活 → 终止状态偏移
- 第 7 维：$\text{softplus}(\cdot)$ 激活 → 轨迹评分

### 3.4 网络参数统计

| 骨干类型 | 参数量 | 推理时间 (CPU) | 推理时间 (GPU) |
|----------|--------|----------------|----------------|
| MLP | ~200K | ~2ms | ~0.5ms |
| Conv1D | ~400K | ~3ms | ~0.8ms |
| ResNet1D | ~2M | ~5ms | ~1.2ms |

---

## 4. 网络输入输出详解

### 4.1 输入数据

#### 4.1.1 激光雷达输入

| 属性 | 值 | 说明 |
|------|-----|------|
| 维度 | $[B, 360]$ | B 为批次大小 |
| 数值范围 | $[0, 1]$ | 归一化后的距离值 |
| 归一化公式 | $(d - d_{min}) / (d_{max} - d_{min})$ | - |
| 物理参数 | $d_{min}=0.1m$, $d_{max}=15m$ | 最小/最大探测距离 |

**预处理流程：**

```
原始距离 [0.1, 15.0] m → 归一化 [0, 1] → 网络输入
```

#### 4.1.2 状态输入

| 属性 | 值 | 说明 |
|------|-----|------|
| 维度 | $[B, N_{prim}, 6]$ | 每个基元独立的状态 |
| 各分量 | $[v_x, v_y, a_x, a_y, g_x, g_y]$ | 速度、加速度、目标位置 |

**状态归一化：**

$$\mathbf{s}_{norm} = \left[\frac{v_x}{v_{max}}, \frac{v_y}{v_{max}}, \frac{a_x}{a_{max}}, \frac{a_y}{a_{max}}, \frac{g_x}{L_g}, \frac{g_y}{L_g}\right]$$

其中：
- $v_{max} = 6.0$ m/s（最大速度）
- $a_{max} = 6.0$ m/s²（最大加速度）
- $L_g = 2 \times r_{radio} = 10.0$ m（目标归一化长度）

**状态变换到各基元坐标系：**

对于第 $j$ 个基元，状态向量需要旋转到该基元的局部坐标系：

$$\mathbf{s}_j^{prim} = R_{body \to prim}^j \cdot \mathbf{s}^{body}$$

### 4.2 输出数据

#### 4.2.1 终止状态 (Endstate)

| 属性 | 值 | 说明 |
|------|-----|------|
| 维度 | $[B, N_{prim}, 6]$ | 每个基元的终止状态 |
| 各分量 | $[p_x, p_y, v_x, v_y, a_x, a_y]$ | 机体坐标系下的位置、速度、加速度 |
| 数值范围 | 物理单位 (m, m/s, m/s²) | 反归一化后 |

**反归一化过程（见 2.2 节）**

#### 4.2.2 轨迹评分 (Score)

| 属性 | 值 | 说明 |
|------|-----|------|
| 维度 | $[B, N_{prim}]$ | 每个基元的代价评分 |
| 激活函数 | softplus | 确保非负 |
| 物理含义 | 轨迹总代价预估 | 越小越优 |

**轨迹选择策略：**

$$j^* = \arg\min_j \text{Score}_j$$

选择评分最低的基元作为最优轨迹。

---

## 5. 五次多项式轨迹

### 5.1 数学形式

五次多项式可以满足 6 个边界条件（起止点的位置、速度、加速度）：

$$p(t) = c_0 + c_1 t + c_2 t^2 + c_3 t^3 + c_4 t^4 + c_5 t^5$$

$$v(t) = \frac{dp}{dt} = c_1 + 2c_2 t + 3c_3 t^2 + 4c_4 t^3 + 5c_5 t^4$$

$$a(t) = \frac{d^2p}{dt^2} = 2c_2 + 6c_3 t + 12c_4 t^2 + 20c_5 t^3$$

$$j(t) = \frac{d^3p}{dt^3} = 6c_3 + 24c_4 t + 60c_5 t^2$$

### 5.2 系数求解

给定边界条件：

$$\begin{cases}
p(0) = p_0, & v(0) = v_0, & a(0) = a_0 \\
p(T) = p_1, & v(T) = v_1, & a(T) = a_1
\end{cases}$$

系数求解：

$$\begin{bmatrix} c_0 \\ c_1 \\ c_2 \\ c_3 \\ c_4 \\ c_5 \end{bmatrix} = A^{-1} \begin{bmatrix} p_0 \\ v_0 \\ a_0 \\ p_1 \\ v_1 \\ a_1 \end{bmatrix}$$

其中 $A^{-1}$ 是系数矩阵的逆：

$$A^{-1} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1/2 & 0 & 0 & 0 \\
-10/T^3 & -6/T^2 & -3/(2T) & 10/T^3 & -4/T^2 & 1/(2T) \\
15/T^4 & 8/T^3 & 3/(2T^2) & -15/T^4 & 7/T^3 & -1/T^2 \\
-6/T^5 & -3/T^4 & -1/(2T^3) & 6/T^5 & -3/T^4 & 1/(2T^3)
\end{bmatrix}$$

### 5.3 轨迹时间

轨迹时间由速度和规划范围决定：

$$T_{segment} = \frac{2 \times r_{radio}}{v_{max}} = \frac{2 \times 5.0}{6.0} \approx 1.67s$$

---

## 6. 损失函数设计

### 6.1 总体损失

$$\mathcal{L}_{total} = w_s \mathcal{L}_{smooth} + w_a \mathcal{L}_{acc} + w_c \mathcal{L}_{safety} + w_g \mathcal{L}_{guidance}$$

默认权重（归一化后）：

| 损失项 | 权重 | 说明 |
|--------|------|------|
| $w_g$ (Guidance) | 0.15 | 目标引导 |
| $w_s$ (Smoothness) | 10.0 / $v_{max}^5$ | 平滑性 (Jerk) |
| $w_a$ (Acceleration) | 1.0 / $v_{max}^3$ | 加速度惩罚 |
| $w_c$ (Safety) | 1.0 | 安全性 |

**权重归一化**：为了在不同速度下保持一致的训练行为，平滑和加速度权重需要按速度缩放。

### 6.2 安全损失 (Safety Loss)

基于 ESDF（欧几里得符号距离场）的碰撞避障损失。

**距离查询**：使用双线性插值从 ESDF 地图获取轨迹点到最近障碍物的距离。

**代价函数**（指数形式，对齐 YOPO_Sim）：

$$c_{safety}(d) = \exp\left(-\frac{d - d_0}{r}\right)$$

其中：
- $d$：到障碍物的距离
- $d_0 = 1.2$ m：安全距离阈值
- $r = 0.6$ m：机器人膨胀半径

**轨迹采样**：使用五次多项式在轨迹上均匀采样 20 个点进行碰撞检测。

$$\mathcal{L}_{safety} = \frac{1}{N_{eval}} \sum_{i=1}^{N_{eval}} c_{safety}(d(\mathbf{p}_i))$$

### 6.3 平滑损失 (Smoothness Loss)

使用 QP 矩阵精确计算 Jerk 积分。

**Jerk 代价**（时间积分）：

$$\mathcal{L}_{jerk} = \int_0^T \|j(t)\|^2 dt = \mathbf{d}^T R_J \mathbf{d}$$

其中 $\mathbf{d} = [p_0, v_0, a_0, p_1, v_1, a_1]^T$ 是边界条件向量，$R_J$ 是 QP 海森矩阵。

**QP 矩阵计算**：

$$H_{jerk}[i,j] = \frac{i(i-1)(i-2) \cdot j(j-1)(j-2)}{i+j-5} T^{i+j-5}, \quad i,j \in [3,5]$$

### 6.4 加速度损失 (Acceleration Loss)

类似于 Jerk 损失，使用 QP 矩阵计算加速度积分：

$$\mathcal{L}_{acc} = \int_0^T \|a(t)\|^2 dt = \mathbf{d}^T R_A \mathbf{d}$$

$$H_{acc}[i,j] = \frac{i(i-1) \cdot j(j-1)}{i+j-3} T^{i+j-3}, \quad i,j \in [2,5]$$

### 6.5 引导损失 (Guidance Loss)

目标引导损失，使轨迹朝向目标方向。

**投影相似度损失**：

$$\mathcal{L}_{guidance} = |g - \mathbf{d}_{traj} \cdot \hat{\mathbf{g}}| + w_{perp} \|\mathbf{d}_{traj}^\perp\|$$

其中：
- $\mathbf{d}_{traj} = \mathbf{p}_{end} - \mathbf{p}_{start}$：轨迹向量
- $\hat{\mathbf{g}} = (\mathbf{g} - \mathbf{p}_{start}) / \|\mathbf{g} - \mathbf{p}_{start}\|$：目标方向单位向量
- $g = \|\mathbf{g} - \mathbf{p}_{start}\|$：到目标的距离
- $\mathbf{d}_{traj}^\perp$：轨迹向量在目标方向上的垂直分量
- $w_{perp} = 0.3$：横向容差权重（越小越允许侧向探索）

**轨迹长度奖励**（避免过短轨迹）：

$$\mathcal{L}_{length} = w_{len} \cdot \max\left(0, 1 - \frac{\|\mathbf{d}_{traj}\|}{2 r_{radio}}\right) \cdot 2 r_{radio}$$

### 6.6 Score 损失

轨迹评分的监督学习损失，使网络预测的评分接近实际轨迹代价：

$$\mathcal{L}_{score} = \text{SmoothL1}(\hat{s}, \mathcal{L}_{total})$$

**总训练损失**：

$$\mathcal{L}_{train} = w_{traj} \cdot \mathcal{L}_{total} + w_{score} \cdot \mathcal{L}_{score}$$

默认权重：$w_{traj} = 1.0$, $w_{score} = 1.0$

---

## 7. 数据集生成

### 7.1 状态采样分布

状态采样参数对齐 YOPO_Sim：

| 变量 | 均值 (归一化) | 标准差 (归一化) | 说明 |
|------|--------------|-----------------|------|
| $v_x$ | 0.4 | 2.0 | 对数正态分布，正偏向 |
| $v_y$ | 0.0 | 0.45 | 正态分布 |
| $a_x$ | 0.0 | 0.5 | 正态分布 |
| $a_y$ | 0.0 | 0.5 | 正态分布 |

**速度采样**：$v_x$ 使用对数正态分布确保正偏向（机器人通常前进）。

### 7.2 目标采样

$$\theta_{goal} \sim \mathcal{N}(0, \sigma_{yaw}^2), \quad \sigma_{yaw} = 20°$$
$$r_{goal} = 2 \times r_{radio} = 10.0m$$
$$\mathbf{g} = \mathbf{p}_{robot} + r_{goal} \cdot [\cos(\theta_{goal}), \sin(\theta_{goal})]^T$$

### 7.3 位置采样策略

为提高训练效果，使用分层采样：

| 策略 | 比例 | 说明 |
|------|------|------|
| 近障碍物 | 30% | 距障碍物 < 2.0m 的位置 |
| 近目标 | 30% | 距目标 < 5.0m（改善最后一公里问题） |
| 随机 | 40% | 地图内随机有效位置 |

### 7.4 数据集格式

| 数据项 | 形状 | 说明 |
|--------|------|------|
| lidar | [N, 360] | 归一化激光数据 |
| positions | [N, 2] | 世界坐标位置 |
| headings | [N] | 朝向角 |
| velocities | [N, 2] | 机体系速度 |
| accelerations | [N, 2] | 机体系加速度 |
| goals | [N, 2] | 世界坐标目标 |
| map_indices | [N] | 地图索引 |

---

## 8. 训练与推理

### 8.1 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Batch Size | 64 | 批次大小 |
| Learning Rate | 1.5e-4 | 初始学习率 |
| Optimizer | AdamW | 权重衰减 1e-4 |
| Epochs | 100 | 训练轮数 |
| Gradient Clipping | 0.1 | 梯度裁剪阈值 |

### 8.2 训练流程

```
1. 加载批次数据 (lidar, position, heading, velocity, acceleration, goal, map_idx)
2. 构建机体系状态: state_body = [vel_body, acc_body, goal_body]
3. 网络推理: endstate, score = network.inference(lidar, state_body)
4. 转换到世界系并计算损失
5. 反向传播 + 梯度裁剪 + 优化器更新
6. 每 save_interval 个 epoch 保存检查点
```

### 8.3 推理流程

```python
# 1. 获取观测
lidar_normalized = lidar.normalize_ranges(lidar_scan)
state_body = robot.get_body_state(goal)

# 2. 网络推理
endstate, score = network.inference(lidar, state_body)

# 3. 碰撞检测过滤
safe_indices = collision_check(endstate, map)

# 4. 选择最优轨迹
best_idx = argmin(score[safe_indices])

# 5. 构建五次多项式轨迹
traj = Poly5Solver2D(pos0, vel0, acc0, pos1, vel1, acc1, T)

# 6. 执行轨迹跟踪
for t in trajectory_time:
    cmd_pos, cmd_vel, cmd_acc = traj.evaluate(t)
    robot.step_with_control(cmd_pos, cmd_vel, cmd_acc, dt)
```

### 8.4 碰撞恢复策略

测试时的轨迹选择包含多重安全机制：

1. **碰撞检测过滤**：对所有候选轨迹进行碰撞检测
2. **综合评分**：结合网络 score、目标距离、轨迹长度
3. **直接到目标**：当距离目标很近时，尝试直接规划到目标
4. **动态权重调整**：根据是否被堵调整评分权重

---

## 9. 仿真环境

### 9.1 地图生成

支持多种地图类型：

| 类型 | 说明 |
|------|------|
| forest | 随机圆形障碍物（森林） |
| maze | 迷宫地图 |
| pillars | 规则柱状障碍物 |
| random | 随机混合障碍物 |

**ESDF 计算**：使用距离变换算法计算欧几里得符号距离场。

### 9.2 激光雷达仿真

| 参数 | 值 | 说明 |
|------|-----|------|
| 光束数 | 360 | 全向雷达 |
| FOV | 360° | 全向覆盖 |
| 最大距离 | 15.0 m | - |
| 最小距离 | 0.1 m | - |
| 噪声标准差 | 0.01 m | 测量噪声 |

**射线投射**：使用步进法在占用栅格上进行射线投射。

### 9.3 机器人动力学

采用简化的二阶积分器模型：

$$\mathbf{p}_{t+1} = \mathbf{p}_t + \mathbf{v}_t \Delta t + \frac{1}{2} \mathbf{a}_t \Delta t^2$$
$$\mathbf{v}_{t+1} = \mathbf{v}_t + \mathbf{a}_t \Delta t$$

**约束**：
- 最大速度：$v_{max} = 6.0$ m/s
- 最大加速度：$a_{max} = 6.0$ m/s²
- 最大角速度：$\omega_{max} = 2.0$ rad/s

---

## 10. 使用指南

### 10.1 安装

```bash
conda create -n yopo2d python=3.8
conda activate yopo2d
cd YOPO_2D
pip install -r requirements.txt
```

### 10.2 训练

```bash
python train.py
```

TensorBoard 日志查看：

```bash
tensorboard --logdir=saved/
```

### 10.3 测试

```bash
# 基本测试
python test.py

# 指定模型
python test.py --model saved/YOPO2D_0/epoch100.pth

# 指定地图类型
python test.py --map_type maze

# 无可视化模式
python test.py --no_vis
```

---

## 11. 关键配置参数

### 11.1 config.yaml 结构

```yaml
# 环境参数
env:
  map_size: [100.0, 100.0]
  resolution: 0.1
  obstacle_num: 80

# 传感器参数
sensor:
  num_beams: 360
  fov: 360.0
  max_range: 15.0

# 机器人参数
robot:
  radius: 0.3
  max_vel: 6.0
  max_acc: 6.0

# 轨迹参数
trajectory:
  horizon_num: 7
  direction_fov: 140.0
  anchor_fov: 30.0
  radio_range: 5.0
  w_guidance: 0.15
  w_smoothness: 10.0
  w_safety: 1.0

# 网络参数
network:
  input_dim: 360
  state_dim: 6
  hidden_dim: 128
  backbone: resnet1d

# 训练参数
training:
  batch_size: 64
  learning_rate: 1.5e-4
  epochs: 100
```

---

## 12. 与 YOPO_Sim (3D版本) 的对齐

### 12.1 主要简化

| 特性 | YOPO_Sim (3D) | YOPO_2D | 说明 |
|------|---------------|---------|------|
| 维度 | 3D | 2D | 去除垂直方向 |
| ROS | 依赖 | 无 | 纯 Python |
| 传感器 | 3D 深度/点云 | 2D 激光 | 简化传感器 |
| 动力学 | 四旋翼 | 二阶积分器 | 简化动力学 |
| CUDA | ESDF 计算 | PyTorch CUDA | GPU加速支持 |

### 12.2 对齐的关键参数

以下参数与 YOPO_Sim 保持一致：

- 状态采样分布 (vx_mean_unit, vx_std_unit, ...)
- 目标采样参数 (goal_yaw_std)
- 损失权重 (w_guidance, w_smoothness, w_safety)
- 安全距离参数 (safe_distance, robot_inflation)
- 网络训练参数 (learning_rate, batch_size)

---

## 13. CUDA 加速模块

YOPO_2D 提供了基于 PyTorch CUDA 的可选加速模块，用于加速计算密集型操作。

### 13.1 加速模块架构

```
simulator/cuda_accelerator.py
├── ESDFCUDA           # ESDF计算（Jump Flooding Algorithm）
├── RaycastCUDA        # 并行射线投射
├── CollisionCheckerCUDA   # 批量碰撞检测
├── TrajectoryGeneratorCUDA  # GPU轨迹生成
└── YOPO2DCUDAAccelerator   # 统一接口
```

### 13.2 ESDF 计算

**Jump Flooding Algorithm (JFA)**：

JFA 是一种适合 GPU 并行的近似距离变换算法。其时间复杂度为 $O(N \cdot \log(max\_dim))$，其中 $N$ 是像素总数。

**算法步骤**：

1. 初始化距离场：障碍物像素距离为 0，自由空间为 $\infty$
2. 对于步长 $k = 2^{n-1}, 2^{n-2}, ..., 1$：
   - 对每个像素，检查 8 个方向偏移 $k$ 的邻居
   - 更新最近障碍物距离
3. 最终距离场为欧几里得距离的近似

**性能特性**：

| 地图尺寸 | CPU (scipy) | GPU (PyTorch) | 加速比 |
|----------|-------------|---------------|--------|
| 500×500 | ~10ms | ~17ms | 0.6x |
| 1000×1000 | ~44ms | ~48ms | 0.9x |
| 2000×2000 | ~238ms | ~236ms | 1.0x |

> **注意**：对于常见尺寸的地图，scipy 的 EDT 实现更快。CUDA ESDF 主要用于需要保持 GPU 张量流水线的场景。

### 13.3 射线投射

**并行步进法**：

所有射线同时在 GPU 上并行步进，无需 Python 循环。

**核心算法**：

```python
# 生成采样距离 [num_steps]
t_values = linspace(min_range, max_range, num_steps)

# 计算所有采样点 [N, num_steps, 2]
points = origin + directions × t_values

# 查询占用状态并找到第一个命中
hit = grid[points] > 0.5
first_hit = cumsum(hit).argmax()
```

**性能**：

- 360 条射线：约 1ms（每条射线 2.8μs）
- 完全并行，无 Python 循环开销

### 13.4 碰撞检测

**基于 ESDF 的快速碰撞查询**：

利用预计算的 ESDF，通过双线性插值查询轨迹点的距离值：

$$d(p) = \text{bilinear\_interp}(ESDF, p / resolution)$$
$$\text{collision} = (d(p) < r_{robot}) \lor \text{out\_of\_bounds}$$

**批量处理**：

- 支持 [B, N, 2] 形状的轨迹批量输入
- 使用 `torch.nn.functional.grid_sample` 进行高效插值
- 64 条轨迹 × 20 点：< 1ms

### 13.5 使用方法

**自动检测与启用**：

```python
from simulator.cuda_accelerator import CUDA_AVAILABLE, get_accelerator

if CUDA_AVAILABLE:
    accelerator = get_accelerator()
    
    # 射线投射
    distances = accelerator.raycast(grid, origin, directions, max_range)
    
    # 碰撞检测
    collision, dist = accelerator.check_collision(esdf, positions, radius, res)
```

**在测试脚本中**：

CUDA 加速会在 `test.py` 中自动检测并启用：

```
[CUDA] PyTorch CUDA available: NVIDIA GeForce RTX 3050
[CUDA] Collision detection accelerator enabled
```

---

## 14. 总结

YOPO_2D 是一个完整的端到端运动规划学习框架，具有以下特点：

1. **理论基础扎实**：基于运动基元 + 五次多项式的轨迹表示
2. **损失设计合理**：多目标损失综合考虑安全、平滑、引导
3. **实现简洁高效**：纯 Python 实现，易于理解和扩展
4. **与原版对齐**：关键参数和算法逻辑与 YOPO_Sim 一致
5. **可选 CUDA 加速**：射线投射和碰撞检测支持 GPU 加速

该项目适合用于：
- 学习 YOPO 算法原理
- 快速验证规划算法
- 2D 导航场景开发
- 算法教学演示
