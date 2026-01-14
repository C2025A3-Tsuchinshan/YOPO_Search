# YOPO 2D

**You Only Plan Once** 的 2D 简化版本 - 纯Python实现，无需ROS依赖。

## 项目简介

这是原始 YOPO_Sim 项目的2D简化版本，主要特点：

- ✅ **纯Python实现** - 无需ROS、C++依赖
- ✅ **2D环境** - 简化的2D导航场景
- ✅ **轻量级** - 可在普通PC上快速训练和测试
- ✅ **易于理解** - 代码结构清晰，适合学习YOPO原理

## 项目结构

```
YOPO_2D/
├── config/
│   ├── __init__.py         # 配置管理器
│   └── config.yaml         # 配置文件
├── simulator/
│   ├── __init__.py
│   ├── map_generator.py    # 2D地图生成器
│   ├── sensor.py           # 2D激光雷达仿真
│   └── dynamics.py         # 机器人动力学模型
├── policy/
│   ├── __init__.py
│   ├── primitive.py        # 2D运动基元
│   ├── network.py          # YOPO神经网络
│   └── loss.py             # 损失函数
├── dataset.py              # 数据集生成和加载
├── train.py                # 训练脚本
├── test.py                 # 测试和可视化
├── requirements.txt        # 依赖列表
└── README.md               # 本文件
```

## 安装

```bash
# 创建虚拟环境
conda create -n yopo2d python=3.8
conda activate yopo2d

# 安装依赖
cd YOPO_2D
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

```bash
python train.py
```

训练日志保存在 `saved/YOPO2D_X/` 目录下。

### 2. 测试模型

```bash
# 使用最新模型
python test.py

# 指定模型路径
python test.py --model saved/YOPO2D_0/epoch100.pth

# 指定地图类型
python test.py --map_type maze

# 无可视化模式
python test.py --no_vis
```

### 3. 查看训练日志

```bash
tensorboard --logdir=saved/
```

## 配置说明

主要配置参数在 `config/config.yaml` 中：

### 环境参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `map_size` | 地图大小 (m) | [100, 100] |
| `resolution` | 地图分辨率 (m) | 0.1 |
| `map_type` | 地图类型 | forest |
| `obstacle_num` | 障碍物数量 | 80 |

### 传感器参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_beams` | 激光束数量 | 360 |
| `fov` | 视场角 (°) | 360 |
| `max_range` | 最大探测距离 (m) | 15 |

### 轨迹参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_directions` | 运动基元数量 | 9 |
| `planning_horizon` | 规划距离 (m) | 4.0 |
| `w_guidance` | 目标引导权重 | 0.2 |
| `w_safety` | 安全性权重 | 2.0 |

## 核心组件

### 1. 地图生成器 (`simulator/map_generator.py`)

支持多种地图类型：
- `forest`: 随机森林（圆形障碍物）
- `maze`: 迷宫
- `pillars`: 柱子（规则排列）

```python
from simulator.map_generator import Map2D

map_2d = Map2D(size=[50, 50], seed=42)
map_2d.generate('forest')
```

### 2. 激光雷达仿真 (`simulator/sensor.py`)

```python
from simulator.sensor import Lidar2D

lidar = Lidar2D(num_beams=360, fov=360, max_range=15)
ranges = lidar.scan(map_2d, position, heading)
```

### 3. 运动基元 (`policy/primitive.py`)

```
基元布局 (以机器人朝向为中心):
    +---+---+---+---+---+---+---+---+---+
    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
    +---+---+---+---+---+---+---+---+---+
   -45°                 0°                +45°
```

### 4. YOPO网络 (`policy/network.py`)

```
输入:
├── 激光数据 [360]
└── 状态 [6]: [vx, vy, ax, ay, goal_x, goal_y]

网络:
├── LiDAR Backbone (MLP/CNN1D) → [hidden_dim]
├── State Encoder (MLP) → [hidden_dim/2]
└── Head → [num_primitives × 7]

输出:
├── endstate [num_primitives, 6]: 终止状态偏移
└── score [num_primitives]: 轨迹评分
```

### 5. 损失函数 (`policy/loss.py`)

- **Safety Loss**: 基于ESDF的避障损失
- **Smoothness Loss**: Jerk最小化（平滑性）
- **Guidance Loss**: 目标引导损失
- **Acceleration Loss**: 加速度惩罚

## 与原始YOPO对比

| 特性 | YOPO (3D) | YOPO 2D |
|------|-----------|---------|
| 维度 | 3D | 2D |
| 传感器 | 深度相机 | 2D激光雷达 |
| 仿真器 | CUDA C++ + ROS | 纯Python |
| 网络输入 | 深度图像 [96×160] | 激光扫描 [360] |
| 运动基元 | 5×3×1 = 15 | 9 |
| 轨迹 | 5次多项式 | 5次多项式 |
| 依赖 | ROS, CUDA, C++ | Python only |

## API 参考

### YopoSimulator2D

```python
from test import YopoSimulator2D

# 创建仿真器
sim = YopoSimulator2D(model_path='path/to/model.pth')

# 重置环境
obs = sim.reset(map_type='forest', seed=42)

# 规划
plan = sim.plan(obs)

# 执行一步
obs, done, info = sim.step(plan)

# 运行完整episode
result = sim.run_episode(visualize=True)
```

### YopoTrainer2D

```python
from train import YopoTrainer2D

# 创建训练器
trainer = YopoTrainer2D(
    learning_rate=1e-4,
    batch_size=64
)

# 设置数据集
trainer.setup_data(num_maps=10, samples_per_map=5000)

# 训练
trainer.train(epochs=100)
```

## 扩展

### 添加新的地图类型

在 `simulator/map_generator.py` 中添加新方法：

```python
def generate_custom_map(self, **kwargs):
    self.grid.fill(0)
    self.obstacles.clear()
    # 你的地图生成逻辑
    self.compute_esdf()
```

### 修改网络结构

在 `policy/network.py` 中修改 `YopoNetwork2D` 类。

## 许可证

MIT License

## 致谢

基于 [YOPO](https://github.com/TJU-Aerial-Robotics/YOPO) 项目简化而来。

```
@article{YOPO,
  title={You Only Plan Once: A Learning-based One-stage Planner with Guidance Learning},
  author={Lu, Junjie and Zhang, Xuewei and Shen, Hongming and Xu, Liwen and Tian, Bailing},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```
