"""
搜索任务数据集 (Search Task Dataset)

与原始 YopoDataset2D 的区别:
- 生成不确定性地图而非目标点
- 支持目标位置模拟 (用于训练时的特权信息)
- 样本包含: lidar, position, heading, velocity, acceleration, map_idx
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from simulator.map_generator import Map2D, MapManager
from simulator.sensor import Lidar2D
from simulator.uncertainty_map import UncertaintyMap, create_search_scenario


class SearchDataset(Dataset):
    """搜索任务数据集"""
    
    def __init__(
        self,
        data_dir: str = None,
        mode: str = 'train',
        val_ratio: float = 0.1,
        num_maps: int = 10,
        samples_per_map: int = 5000,
        num_targets: int = 1
    ):
        """
        Args:
            data_dir: 数据目录
            mode: 'train' 或 'valid'
            val_ratio: 验证集比例
            num_maps: 地图数量
            samples_per_map: 每个地图的样本数量
            num_targets: 每个地图的目标数量
        """
        super().__init__()
        
        self.mode = mode
        self.val_ratio = val_ratio
        self.num_targets = num_targets
        
        # 参数 (对齐 YOPO_Sim 配置)
        train_cfg = cfg['training']
        robot_cfg = cfg['robot']
        search_cfg = cfg.get('search', default={}) if isinstance(cfg._data, dict) else {}
        
        self.max_vel = float(train_cfg.get('vel_max_train', robot_cfg['max_vel']))
        self.max_acc = float(train_cfg.get('acc_max_train', robot_cfg['max_acc']))
        
        # 状态采样参数
        self.vx_mean_unit = float(train_cfg.get('vx_mean_unit', 0.4))
        self.vx_std_unit = float(train_cfg.get('vx_std_unit', 2.0))
        self.vy_mean_unit = float(train_cfg.get('vy_mean_unit', 0.0))
        self.vy_std_unit = float(train_cfg.get('vy_std_unit', 0.45))
        self.ax_mean_unit = float(train_cfg.get('ax_mean_unit', 0.0))
        self.ax_std_unit = float(train_cfg.get('ax_std_unit', 0.5))
        self.ay_mean_unit = float(train_cfg.get('ay_mean_unit', 0.0))
        self.ay_std_unit = float(train_cfg.get('ay_std_unit', 0.5))
        
        self.v_mean = np.array([self.vx_mean_unit, self.vy_mean_unit])
        self.v_std = np.array([self.vx_std_unit, self.vy_std_unit])
        self.a_mean = np.array([self.ax_mean_unit, self.ay_mean_unit])
        self.a_std = np.array([self.ax_std_unit, self.ay_std_unit])
        
        # 对数正态分布参数 (用于 vx 采样)
        self.vx_lognorm_mean = np.log(max(1e-3, 1.0 - self.vx_mean_unit))
        self.vx_lognorm_sigma = np.log(max(1e-3, self.vx_std_unit))
        
        # 近障碍物采样
        self.resample_state = bool(train_cfg.get('resample_state', True))
        self.near_obstacle_ratio = float(train_cfg.get('near_obstacle_ratio', 0.2))
        self.near_obstacle_distance = float(
            train_cfg.get('near_obstacle_distance', cfg['trajectory']['safe_distance'])
        )
        
        # 数据存储
        self.lidar_data = []
        self.positions = []
        self.headings = []
        self.velocities = []
        self.accelerations = []
        self.map_indices = []
        
        # ESDF地图和不确定性地图
        self.esdf_maps = []
        self.uncertainty_maps = []
        self.target_positions = []  # 每个地图的目标位置
        
        self.map_info = {
            'size': cfg['env']['map_size'],
            'resolution': cfg['env']['resolution']
        }
        
        # 传感器
        self.lidar = Lidar2D()
        
        # 生成数据
        self._generate_data(num_maps, samples_per_map)
        
        # 划分训练/验证集
        self._split_data()
        
        print(f"SearchDataset [{mode}]: {len(self)} samples, {len(self.esdf_maps)} maps")
    
    def _generate_data(self, num_maps: int, samples_per_map: int):
        """生成数据"""
        print(f"Generating search dataset with {num_maps} maps, {samples_per_map} samples each...")
        
        robot_radius = cfg['robot']['radius']
        search_cfg = cfg.get('search', default={}) if isinstance(cfg._data, dict) else {}
        
        for map_idx in range(num_maps):
            print(f"  Generating map {map_idx + 1}/{num_maps}...")
            
            # 生成障碍物地图
            map_2d = Map2D(seed=cfg['env']['seed'] + map_idx)
            map_2d.generate()
            
            # 保存ESDF
            self.esdf_maps.append(map_2d.esdf.copy())
            
            # 生成不确定性地图和目标位置
            uncertainty_map = UncertaintyMap(
                map_size=self.map_info['size'],
                resolution=self.map_info['resolution'],
                initial_prob=float(search_cfg.get('initial_prob', 0.5)),
                detection_prob=float(search_cfg.get('detection_prob', 0.9)),
                false_alarm_prob=float(search_cfg.get('false_alarm_prob', 0.1)),
                k_eta=float(search_cfg.get('k_eta', 1.0)),
                sensor_radius=float(search_cfg.get('sensor_radius', 5.0))
            )
            
            # 随机放置目标 (在自由空间)
            target_pos = self._sample_target_position(map_2d, robot_radius)
            self.target_positions.append(target_pos)
            
            # 模拟一些随机探索更新不确定性地图
            self._simulate_exploration(uncertainty_map, map_2d, target_pos, robot_radius)
            
            # 保存不确定性地图
            uncertainty_map.compute_uncertainty()
            self.uncertainty_maps.append(uncertainty_map.uncertainty_map.copy())
            
            # 采样
            for _ in range(samples_per_map):
                # 采样有效位置
                pos = self._sample_position(map_2d, robot_radius)
                if pos is None:
                    continue
                
                # 随机朝向
                heading = np.random.uniform(-np.pi, np.pi)
                
                # 执行激光扫描
                lidar_scan = self.lidar.scan(map_2d, pos, heading, add_noise=True)
                lidar_normalized = self.lidar.normalize_ranges(lidar_scan)
                
                # 随机速度和加速度 (机体系)
                vel_body = self._sample_velocity()
                acc_body = self._sample_acceleration()
                
                # 保存数据
                self.lidar_data.append(lidar_normalized)
                self.positions.append(pos)
                self.headings.append(heading)
                self.velocities.append(vel_body)
                self.accelerations.append(acc_body)
                self.map_indices.append(map_idx)
        
        # 转换为numpy数组
        self.lidar_data = np.array(self.lidar_data, dtype=np.float32)
        self.positions = np.array(self.positions, dtype=np.float32)
        self.headings = np.array(self.headings, dtype=np.float32)
        self.velocities = np.array(self.velocities, dtype=np.float32)
        self.accelerations = np.array(self.accelerations, dtype=np.float32)
        self.map_indices = np.array(self.map_indices, dtype=np.int64)
        
        print(f"Generated {len(self.lidar_data)} samples")
    
    def _sample_target_position(self, map_2d: Map2D, robot_radius: float) -> np.ndarray:
        """采样目标位置 (在自由空间)"""
        margin = min(self.map_info['size']) * 0.1
        
        for _ in range(100):
            pos = np.random.uniform(
                [margin, margin],
                [self.map_info['size'][0] - margin, self.map_info['size'][1] - margin]
            )
            if map_2d.get_distance(pos) > robot_radius * 2:
                return pos.astype(np.float32)
        
        # 回退到随机位置
        return np.random.uniform(
            [margin, margin],
            [self.map_info['size'][0] - margin, self.map_info['size'][1] - margin]
        ).astype(np.float32)
    
    def _simulate_exploration(
        self,
        uncertainty_map: UncertaintyMap,
        map_2d: Map2D,
        target_pos: np.ndarray,
        robot_radius: float,
        num_steps: int = 50
    ):
        """
        模拟探索过程，更新不确定性地图
        
        这会在地图上留下一些已探索的区域 (低不确定性)
        """
        # 随机游走探索
        pos = map_2d.sample_free_position(robot_radius)
        if pos is None:
            return
        
        for _ in range(num_steps):
            # 更新不确定性
            uncertainty_map.update_observation(
                pos, detected=False, target_position=target_pos
            )
            
            # 随机移动
            direction = np.random.randn(2)
            direction = direction / (np.linalg.norm(direction) + 1e-6) * 3.0
            new_pos = pos + direction
            
            # 检查是否在有效范围内
            if (0 < new_pos[0] < self.map_info['size'][0] and
                0 < new_pos[1] < self.map_info['size'][1] and
                map_2d.get_distance(new_pos) > robot_radius):
                pos = new_pos
    
    def _sample_position(self, map_2d: Map2D, robot_radius: float) -> Optional[np.ndarray]:
        """采样位置"""
        use_near = np.random.rand() < self.near_obstacle_ratio
        max_attempts = 200
        
        for _ in range(max_attempts):
            pos = map_2d.sample_free_position(robot_radius)
            if pos is None:
                continue
            if self.near_obstacle_ratio <= 0.0:
                return pos
            dist = map_2d.get_distance(pos)
            if use_near:
                if dist > robot_radius and dist <= self.near_obstacle_distance:
                    return pos
            else:
                if dist > self.near_obstacle_distance:
                    return pos
        
        return map_2d.sample_free_position(robot_radius)
    
    def _sample_velocity(self) -> np.ndarray:
        """采样速度 (机体系)"""
        while True:
            vel = self.max_vel * (self.v_mean + self.v_std * np.random.randn(2))
            right_skewed_vx = -1.0
            while right_skewed_vx < 0:
                right_skewed_vx = self.max_vel * np.random.lognormal(
                    mean=self.vx_lognorm_mean,
                    sigma=self.vx_lognorm_sigma
                )
                right_skewed_vx = -right_skewed_vx + 1.2 * self.max_vel
            vel[0] = right_skewed_vx
            if np.linalg.norm(vel) < 1.2 * self.max_vel:
                break
        return vel.astype(np.float32)
    
    def _sample_acceleration(self) -> np.ndarray:
        """采样加速度 (机体系)"""
        while True:
            acc = self.max_acc * (self.a_mean + self.a_std * np.random.randn(2))
            if np.linalg.norm(acc) < 1.2 * self.max_acc:
                break
        return acc.astype(np.float32)
    
    def _split_data(self):
        """划分训练/验证集"""
        n_total = len(self.lidar_data)
        n_val = int(n_total * self.val_ratio)
        
        indices = np.random.permutation(n_total)
        
        if self.mode == 'train':
            self.indices = indices[n_val:]
        else:
            self.indices = indices[:n_val]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        """
        获取样本
        
        Returns:
            lidar: [num_beams] 归一化激光数据
            position: [2] 世界坐标
            heading: scalar 朝向
            velocity: [2] 机体速度
            acceleration: [2] 机体加速度
            map_idx: scalar 地图索引
        """
        i = self.indices[idx]
        
        if self.resample_state:
            vel = self._sample_velocity()
            acc = self._sample_acceleration()
        else:
            vel = self.velocities[i]
            acc = self.accelerations[i]
        
        return (
            torch.tensor(self.lidar_data[i]),
            torch.tensor(self.positions[i]),
            torch.tensor(self.headings[i]),
            torch.tensor(vel),
            torch.tensor(acc),
            torch.tensor(self.map_indices[i])
        )
    
    def get_target_position(self, map_idx: int) -> np.ndarray:
        """获取指定地图的目标位置"""
        return self.target_positions[map_idx]


def generate_search_dataset(
    save_dir: str = None,
    num_maps: int = 10,
    samples_per_map: int = 5000,
    num_targets: int = 1
) -> SearchDataset:
    """生成并保存搜索数据集"""
    dataset = SearchDataset(
        data_dir=None,
        mode='train',
        num_maps=num_maps,
        samples_per_map=samples_per_map,
        num_targets=num_targets
    )
    
    return dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Testing search dataset...")
    
    dataset = SearchDataset(
        mode='train',
        num_maps=2,
        samples_per_map=100
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of maps: {len(dataset.esdf_maps)}")
    print(f"Number of uncertainty maps: {len(dataset.uncertainty_maps)}")
    
    # 获取样本
    sample = dataset[0]
    print(f"Sample shapes: {[s.shape for s in sample]}")
    
    # 可视化第一个地图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    ax.imshow(dataset.esdf_maps[0].T, origin='lower', cmap='viridis')
    target = dataset.target_positions[0]
    ax.plot(target[0] / cfg['env']['resolution'], target[1] / cfg['env']['resolution'], 
            'r*', markersize=15)
    ax.set_title('ESDF Map')
    
    ax = axes[1]
    ax.imshow(dataset.uncertainty_maps[0].T, origin='lower', cmap='hot', vmin=0, vmax=1)
    ax.plot(target[0] / cfg['env']['resolution'], target[1] / cfg['env']['resolution'], 
            'c*', markersize=15)
    ax.set_title('Uncertainty Map')
    
    ax = axes[2]
    # 显示一些采样位置
    map_0_indices = np.where(dataset.map_indices == 0)[0][:50]
    positions = dataset.positions[map_0_indices]
    ax.scatter(positions[:, 0] / cfg['env']['resolution'], 
              positions[:, 1] / cfg['env']['resolution'], 
              c='blue', s=10, alpha=0.5)
    ax.set_title('Sample Positions')
    ax.set_xlim(0, dataset.uncertainty_maps[0].shape[0])
    ax.set_ylim(0, dataset.uncertainty_maps[0].shape[1])
    
    plt.tight_layout()
    plt.savefig('search_dataset_test.png')
    print("Saved to search_dataset_test.png")
    plt.show()
