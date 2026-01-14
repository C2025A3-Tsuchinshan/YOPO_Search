"""
不确定性地图模块 (Uncertainty Map)

用于搜索任务的认知地图，管理目标存在概率和不确定性。

核心概念:
- p_m: 网格 m 中目标存在的概率 (0~1)
- Q_m: 对数变换后的概率 Q = ln(1/p - 1)
- η_m: 不确定性 η = exp(-K_η * |Q|)

参考 test2.md 方案的贝叶斯更新公式。
"""

import numpy as np
import torch
from typing import Optional, Tuple, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg


class UncertaintyMap:
    """
    不确定性地图类
    
    管理目标存在概率的贝叶斯更新和不确定性计算。
    """
    
    def __init__(
        self,
        map_size: Tuple[float, float] = None,
        resolution: float = None,
        initial_prob: float = 0.5,
        detection_prob: float = 0.9,
        false_alarm_prob: float = 0.1,
        k_eta: float = 1.0,
        sensor_radius: float = 5.0,
        device: str = None
    ):
        """
        Args:
            map_size: 地图大小 [x, y] (meters)
            resolution: 网格分辨率 (meters/cell)
            initial_prob: 初始目标存在概率
            detection_prob: 检测概率 p_d
            false_alarm_prob: 虚警概率 p_f
            k_eta: 不确定性增益参数
            sensor_radius: 传感器覆盖半径
            device: 计算设备
        """
        env_cfg = cfg['env']
        self.map_size = map_size or env_cfg['map_size']
        self.resolution = resolution or env_cfg['resolution']
        
        # 网格尺寸
        self.grid_size = (
            int(self.map_size[0] / self.resolution),
            int(self.map_size[1] / self.resolution)
        )
        
        # 贝叶斯参数
        self.initial_prob = initial_prob
        self.p_d = detection_prob  # 检测概率
        self.p_f = false_alarm_prob  # 虚警概率
        self.k_eta = k_eta  # 不确定性增益
        self.sensor_radius = sensor_radius  # 传感器覆盖半径
        
        # 设备
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 初始化概率图
        self._init_probability_map()
        
        # 预计算更新常数 (对数变换)
        self._compute_update_constants()
    
    def _init_probability_map(self):
        """初始化概率图"""
        # p_m: 目标存在概率
        self.prob_map = np.full(self.grid_size, self.initial_prob, dtype=np.float32)
        
        # Q_m: 对数变换后的概率 (用于线性更新)
        # Q = ln(1/p - 1)
        self.log_odds_map = np.full(self.grid_size, 0.0, dtype=np.float32)  # p=0.5 时 Q=0
        
        # η_m: 不确定性地图
        self.uncertainty_map = np.ones(self.grid_size, dtype=np.float32)  # 初始最大不确定性
        
        # 访问计数
        self.visit_count = np.zeros(self.grid_size, dtype=np.int32)
    
    def _compute_update_constants(self):
        """预计算贝叶斯更新常数"""
        # 检测到目标时的更新量 v = ln(p_f / p_d)
        self.v_detect = np.log(self.p_f / self.p_d)
        
        # 未检测到目标时的更新量 v = ln((1-p_f) / (1-p_d))
        self.v_no_detect = np.log((1 - self.p_f) / (1 - self.p_d))
    
    def reset(self, initial_prob: float = None):
        """重置地图"""
        self.initial_prob = initial_prob or self.initial_prob
        self._init_probability_map()
    
    def world_to_grid(self, position: np.ndarray) -> np.ndarray:
        """世界坐标到网格坐标"""
        position = np.asarray(position)
        grid_pos = (position / self.resolution).astype(np.int32)
        grid_pos = np.clip(grid_pos, 0, np.array(self.grid_size) - 1)
        return grid_pos
    
    def grid_to_world(self, grid_pos: np.ndarray) -> np.ndarray:
        """网格坐标到世界坐标 (网格中心)"""
        return (grid_pos + 0.5) * self.resolution
    
    def get_covered_cells(self, position: np.ndarray) -> List[Tuple[int, int]]:
        """
        获取传感器覆盖的网格列表
        
        Args:
            position: 传感器位置 [x, y]
            
        Returns:
            cells: 覆盖的网格坐标列表
        """
        center = self.world_to_grid(position)
        radius_cells = int(np.ceil(self.sensor_radius / self.resolution))
        
        cells = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                gx, gy = center[0] + dx, center[1] + dy
                if 0 <= gx < self.grid_size[0] and 0 <= gy < self.grid_size[1]:
                    # 检查是否在圆形范围内
                    cell_world = self.grid_to_world(np.array([gx, gy]))
                    if np.linalg.norm(cell_world - position) <= self.sensor_radius:
                        cells.append((gx, gy))
        return cells
    
    def update_observation(
        self,
        position: np.ndarray,
        detected: bool = False,
        target_position: np.ndarray = None
    ):
        """
        基于观测更新概率图 (贝叶斯更新)
        
        Args:
            position: 传感器位置
            detected: 是否检测到目标
            target_position: 真实目标位置 (用于模拟, 可选)
        """
        covered_cells = self.get_covered_cells(position)
        
        for gx, gy in covered_cells:
            # 更新访问计数
            self.visit_count[gx, gy] += 1
            
            # 确定观测结果
            if target_position is not None:
                # 模拟模式: 根据真实目标位置和检测概率
                cell_center = self.grid_to_world(np.array([gx, gy]))
                has_target = np.linalg.norm(cell_center - target_position) <= self.resolution
                if has_target:
                    observed = np.random.rand() < self.p_d
                else:
                    observed = np.random.rand() < self.p_f
            else:
                observed = detected
            
            # 对数域更新 (线性)
            if observed:
                self.log_odds_map[gx, gy] += self.v_detect
            else:
                self.log_odds_map[gx, gy] += self.v_no_detect
            
            # 限制范围避免数值问题
            self.log_odds_map[gx, gy] = np.clip(self.log_odds_map[gx, gy], -10, 10)
    
    def update_from_trajectory(
        self,
        trajectory: np.ndarray,
        target_position: np.ndarray = None
    ):
        """
        基于轨迹更新概率图
        
        Args:
            trajectory: 轨迹点 [N, 2]
            target_position: 目标位置 (用于模拟)
        """
        for pos in trajectory:
            self.update_observation(pos, detected=False, target_position=target_position)
    
    def compute_probability(self):
        """从对数域计算概率图"""
        # p = 1 / (1 + exp(Q))
        self.prob_map = 1.0 / (1.0 + np.exp(self.log_odds_map))
        return self.prob_map
    
    def compute_uncertainty(self):
        """
        计算不确定性地图
        
        η = exp(-K_η * |Q|)
        Q 越接近 0 (p 越接近 0.5), 不确定性越高
        """
        self.uncertainty_map = np.exp(-self.k_eta * np.abs(self.log_odds_map))
        return self.uncertainty_map
    
    def get_uncertainty(self, position: np.ndarray) -> float:
        """查询位置的不确定性"""
        grid_pos = self.world_to_grid(position)
        return self.uncertainty_map[grid_pos[0], grid_pos[1]]
    
    def get_probability(self, position: np.ndarray) -> float:
        """查询位置的目标存在概率"""
        grid_pos = self.world_to_grid(position)
        return self.prob_map[grid_pos[0], grid_pos[1]]
    
    def query_uncertainty_batch(
        self,
        positions: np.ndarray
    ) -> np.ndarray:
        """
        批量查询不确定性 (支持双线性插值)
        
        Args:
            positions: [N, 2] 或 [batch, N, 2] 世界坐标
            
        Returns:
            uncertainties: 对应形状的不确定性值
        """
        original_shape = positions.shape[:-1]
        positions = positions.reshape(-1, 2)
        
        # 确保不确定性地图已更新
        self.compute_uncertainty()
        
        # 网格坐标 (浮点数, 用于插值)
        grid_x = positions[:, 0] / self.resolution
        grid_y = positions[:, 1] / self.resolution
        
        # 限制范围
        grid_x = np.clip(grid_x, 0, self.grid_size[0] - 1.001)
        grid_y = np.clip(grid_y, 0, self.grid_size[1] - 1.001)
        
        # 双线性插值
        x0 = np.floor(grid_x).astype(np.int32)
        y0 = np.floor(grid_y).astype(np.int32)
        x1 = np.minimum(x0 + 1, self.grid_size[0] - 1)
        y1 = np.minimum(y0 + 1, self.grid_size[1] - 1)
        
        wx = grid_x - x0
        wy = grid_y - y0
        
        v00 = self.uncertainty_map[x0, y0]
        v10 = self.uncertainty_map[x1, y0]
        v01 = self.uncertainty_map[x0, y1]
        v11 = self.uncertainty_map[x1, y1]
        
        values = (
            (1 - wx) * (1 - wy) * v00 +
            wx * (1 - wy) * v10 +
            (1 - wx) * wy * v01 +
            wx * wy * v11
        )
        
        return values.reshape(original_shape)
    
    def to_tensor(self, device: str = None) -> torch.Tensor:
        """转换为 PyTorch 张量"""
        device = device or self.device
        self.compute_uncertainty()
        return torch.tensor(self.uncertainty_map, dtype=torch.float32, device=device)
    
    def get_high_uncertainty_positions(
        self,
        threshold: float = 0.8,
        max_count: int = 10
    ) -> np.ndarray:
        """
        获取高不确定性区域的位置
        
        Args:
            threshold: 不确定性阈值
            max_count: 最大返回数量
            
        Returns:
            positions: [N, 2] 高不确定性位置
        """
        self.compute_uncertainty()
        high_uncertainty = self.uncertainty_map >= threshold
        
        indices = np.argwhere(high_uncertainty)
        if len(indices) == 0:
            return np.array([])
        
        # 按不确定性排序
        uncertainties = self.uncertainty_map[indices[:, 0], indices[:, 1]]
        sorted_idx = np.argsort(-uncertainties)[:max_count]
        
        positions = self.grid_to_world(indices[sorted_idx])
        return positions
    
    def place_target(self, position: np.ndarray):
        """
        在指定位置放置目标 (用于模拟)
        
        这会在目标位置设置高概率, 用于训练时的特权信息
        """
        grid_pos = self.world_to_grid(position)
        # 设置极高概率 (接近 1)
        self.log_odds_map[grid_pos[0], grid_pos[1]] = -5  # p ≈ 0.993
        self.compute_probability()
        self.compute_uncertainty()


class UncertaintyMapTensor:
    """
    GPU 加速的不确定性地图 (用于训练)
    
    支持批量操作和自动微分
    """
    
    def __init__(
        self,
        uncertainty_maps: List[np.ndarray],
        map_info: dict = None,
        device: str = None
    ):
        """
        Args:
            uncertainty_maps: 不确定性地图列表 [H, W] 或 numpy 数组
            map_info: 地图信息 {'size': [x, y], 'resolution': r}
            device: 计算设备
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 转换为张量
        if uncertainty_maps:
            self.maps = torch.stack([
                torch.tensor(m, dtype=torch.float32, device=self.device)
                for m in uncertainty_maps
            ], dim=0).unsqueeze(1)  # [M, 1, H, W]
        else:
            self.maps = None
        
        self.map_info = map_info or {
            'size': cfg['env']['map_size'],
            'resolution': cfg['env']['resolution']
        }
        self.resolution = self.map_info['resolution']
    
    def query_uncertainty(
        self,
        positions: torch.Tensor,
        map_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        批量查询不确定性 (可微分)
        
        Args:
            positions: [batch, N, 2] 世界坐标
            map_idx: [batch] 地图索引
            
        Returns:
            uncertainties: [batch, N]
        """
        if self.maps is None:
            # 默认返回均匀不确定性
            return torch.ones(
                positions.shape[0], positions.shape[1],
                device=self.device
            )
        
        batch_size, N, _ = positions.shape
        
        # 归一化到 [-1, 1] 用于 grid_sample
        h, w = self.maps.shape[2], self.maps.shape[3]
        grid_x = 2.0 * positions[:, :, 0] / (self.map_info['size'][0]) - 1.0
        grid_y = 2.0 * positions[:, :, 1] / (self.map_info['size'][1]) - 1.0
        
        grid = torch.stack([grid_y, grid_x], dim=-1)  # [batch, N, 2]
        grid = grid.unsqueeze(1)  # [batch, 1, N, 2]
        
        # 每个样本选择对应的地图
        map_count = self.maps.shape[0]
        map_idx = map_idx % map_count
        
        uncertainties = []
        for i in range(batch_size):
            m = self.maps[map_idx[i]]  # [1, H, W]
            u = torch.nn.functional.grid_sample(
                m.unsqueeze(0), grid[i:i+1],
                mode='bilinear', padding_mode='border', align_corners=True
            )
            uncertainties.append(u.squeeze())
        
        return torch.stack(uncertainties, dim=0)  # [batch, N]


def create_search_scenario(
    map_size: Tuple[float, float] = None,
    resolution: float = None,
    num_targets: int = 1,
    seed: int = None
) -> Tuple[UncertaintyMap, np.ndarray]:
    """
    创建搜索场景
    
    Args:
        map_size: 地图大小
        resolution: 分辨率
        num_targets: 目标数量
        seed: 随机种子
        
    Returns:
        uncertainty_map: 不确定性地图
        target_positions: 目标位置数组 [num_targets, 2]
    """
    if seed is not None:
        np.random.seed(seed)
    
    env_cfg = cfg['env']
    map_size = map_size or env_cfg['map_size']
    resolution = resolution or env_cfg['resolution']
    
    # 创建不确定性地图
    uncertainty_map = UncertaintyMap(
        map_size=map_size,
        resolution=resolution,
        initial_prob=0.5
    )
    
    # 随机放置目标 (在地图边缘留出边距)
    margin = min(map_size) * 0.1
    target_positions = np.random.uniform(
        [margin, margin],
        [map_size[0] - margin, map_size[1] - margin],
        size=(num_targets, 2)
    ).astype(np.float32)
    
    return uncertainty_map, target_positions


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 测试不确定性地图
    print("Testing uncertainty map...")
    
    um, targets = create_search_scenario(
        map_size=[50, 50],
        resolution=0.5,
        num_targets=1,
        seed=42
    )
    
    print(f"Grid size: {um.grid_size}")
    print(f"Target position: {targets[0]}")
    
    # 模拟搜索过程
    trajectory = np.array([
        [10, 10], [15, 15], [20, 20], [25, 25], [30, 30]
    ])
    
    for pos in trajectory:
        um.update_observation(pos, detected=False, target_position=targets[0])
    
    # 计算并可视化
    um.compute_probability()
    um.compute_uncertainty()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 概率图
    ax = axes[0]
    im = ax.imshow(um.prob_map.T, origin='lower', cmap='RdYlGn',
                   extent=[0, um.map_size[0], 0, um.map_size[1]], vmin=0, vmax=1)
    ax.plot(targets[0, 0], targets[0, 1], 'r*', markersize=15, label='Target')
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', label='Trajectory')
    ax.set_title('Target Probability p(θ=1)')
    ax.legend()
    plt.colorbar(im, ax=ax)
    
    # 对数概率图
    ax = axes[1]
    im = ax.imshow(um.log_odds_map.T, origin='lower', cmap='coolwarm',
                   extent=[0, um.map_size[0], 0, um.map_size[1]])
    ax.plot(targets[0, 0], targets[0, 1], 'r*', markersize=15)
    ax.set_title('Log-Odds Q')
    plt.colorbar(im, ax=ax)
    
    # 不确定性图
    ax = axes[2]
    im = ax.imshow(um.uncertainty_map.T, origin='lower', cmap='hot',
                   extent=[0, um.map_size[0], 0, um.map_size[1]], vmin=0, vmax=1)
    ax.plot(targets[0, 0], targets[0, 1], 'r*', markersize=15)
    ax.set_title('Uncertainty η')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('uncertainty_map_test.png')
    print("Saved to uncertainty_map_test.png")
    plt.show()
