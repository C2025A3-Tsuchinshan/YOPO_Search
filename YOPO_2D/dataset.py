"""
2D YOPO 数据集
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from simulator.map_generator import Map2D, MapManager
from simulator.sensor import Lidar2D


class YopoDataset2D(Dataset):
    """2D YOPO数据集"""
    
    def __init__(
        self,
        data_dir: str = None,
        mode: str = 'train',
        val_ratio: float = 0.1,
        num_maps: int = 10,
        samples_per_map: int = 5000
    ):
        """
        Args:
            data_dir: 数据目录 (如果存在则加载, 否则生成)
            mode: 'train' 或 'valid'
            val_ratio: 验证集比例
            num_maps: 地图数量
            samples_per_map: 每个地图的样本数量
        """
        super().__init__()
        
        self.mode = mode
        self.val_ratio = val_ratio
        
        # 参数 (对齐 YOPO_Sim 配置)
        train_cfg = cfg['training']
        robot_cfg = cfg['robot']
        traj_cfg = cfg['trajectory']
        
        self.max_vel = float(train_cfg.get('vel_max_train', robot_cfg['max_vel']))
        self.max_acc = float(train_cfg.get('acc_max_train', robot_cfg['max_acc']))
        
        # 状态采样参数 (对齐 YOPO_Sim)
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
        
        # 目标采样参数 (对齐 YOPO_Sim)
        radio_range = float(traj_cfg.get('radio_range', traj_cfg['planning_horizon'] / 2))
        self.goal_length = float(train_cfg.get('goal_length', 2.0 * radio_range))
        self.goal_yaw_std = np.deg2rad(float(train_cfg.get('goal_yaw_std', 20.0)))
        self.near_goal_ratio = float(train_cfg.get('near_goal_ratio', 0.3))
        self.near_goal_distance = float(train_cfg.get('near_goal_distance', 5.0))
        
        # 近障碍物采样
        self.resample_state = bool(train_cfg.get('resample_state', True))
        self.near_obstacle_ratio = float(train_cfg.get('near_obstacle_ratio', 0.2))
        self.near_obstacle_distance = float(
            train_cfg.get('near_obstacle_distance', traj_cfg['safe_distance'])
        )
        
        # 数据
        self.lidar_data = []
        self.positions = []
        self.headings = []
        self.velocities = []
        self.accelerations = []
        self.goals = []
        self.map_indices = []
        
        # ESDF地图 (用于损失计算)
        self.esdf_maps = []
        self.map_info = {
            'size': cfg['env']['map_size'],
            'resolution': cfg['env']['resolution']
        }
        
        # 传感器
        self.lidar = Lidar2D()
        
        # 加载或生成数据
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'dataset')
        os.makedirs(self.data_dir, exist_ok=True)
        if data_dir and os.path.exists(data_dir):
            self._load_data(data_dir)
        else:
            self._generate_data(num_maps, samples_per_map)
        
        # 划分训练/验证集
        self._split_data()
        
        print(f"Dataset [{mode}]: {len(self)} samples")
    
    def _generate_data(self, num_maps: int, samples_per_map: int):
        """生成数据"""
        print(f"Generating dataset with {num_maps} maps, {samples_per_map} samples each...")
        
        robot_radius = cfg['robot']['radius']
        
        for map_idx in range(num_maps):
            print(f"  Generating map {map_idx + 1}/{num_maps}...")
            
            # 生成地图
            map_2d = Map2D(seed=cfg['env']['seed'] + map_idx)
            map_2d.generate()
            
            # 保存ESDF
            self.esdf_maps.append(map_2d.esdf.copy())
            
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
                
                # 随机目标 (世界系)
                goal = self._sample_goal(pos, heading)
                
                # 保存数据
                self.lidar_data.append(lidar_normalized)
                self.positions.append(pos)
                self.headings.append(heading)
                self.velocities.append(vel_body)
                self.accelerations.append(acc_body)
                self.goals.append(goal)
                self.map_indices.append(map_idx)
        
        # 转换为numpy数组
        self.lidar_data = np.array(self.lidar_data, dtype=np.float32)
        self.positions = np.array(self.positions, dtype=np.float32)
        self.headings = np.array(self.headings, dtype=np.float32)
        self.velocities = np.array(self.velocities, dtype=np.float32)
        self.accelerations = np.array(self.accelerations, dtype=np.float32)
        self.goals = np.array(self.goals, dtype=np.float32)
        self.map_indices = np.array(self.map_indices, dtype=np.int64)
        
        print(f"Generated {len(self.lidar_data)} samples")

    def _sample_position(self, map_2d: Map2D, robot_radius: float) -> Optional[np.ndarray]:
        """Sample positions with a near-obstacle bias similar to YOPO_Sim."""
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
        # X方向使用正偏分布
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
    
    def _sample_goal(self, pos: np.ndarray, heading: float) -> np.ndarray:
        """
        采样目标点 (世界系) - 对齐 YOPO_Sim
        
        使用 goal_yaw_std 控制目标偏航角，goal_length 控制目标距离
        near_goal_ratio 概率下采样近距离目标（改善最后一公里问题）
        """
        goal_yaw = np.random.normal(0.0, self.goal_yaw_std)
        angle = heading + goal_yaw
        
        # 距离缩放：near_goal_ratio 概率采样近目标 (使用 near_goal_distance)
        random_val = np.random.rand()
        if random_val < self.near_goal_ratio:
            # 近目标：距离在 [robot_radius, near_goal_distance] 范围内
            scale = random_val / max(self.near_goal_ratio, 1e-6)  # [0, 1]
            goal_dist = scale * self.near_goal_distance
        else:
            # 远目标：使用标准 goal_length
            goal_dist = self.goal_length
        
        goal = pos + goal_dist * np.array([np.cos(angle), np.sin(angle)])
        return goal.astype(np.float32)
    
    def _load_data(self, data_dir: str):
        """加载数据"""
        print(f"Loading dataset from {data_dir}...")
        
        self.lidar_data = np.load(os.path.join(data_dir, 'lidar.npy'))
        self.positions = np.load(os.path.join(data_dir, 'positions.npy'))
        self.headings = np.load(os.path.join(data_dir, 'headings.npy'))
        self.velocities = np.load(os.path.join(data_dir, 'velocities.npy'))
        self.accelerations = np.load(os.path.join(data_dir, 'accelerations.npy'))
        self.goals = np.load(os.path.join(data_dir, 'goals.npy'))
        self.map_indices = np.load(os.path.join(data_dir, 'map_indices.npy'))
        
        # 加载ESDF
        esdf_files = sorted([f for f in os.listdir(data_dir) if f.startswith('esdf_')])
        for f in esdf_files:
            self.esdf_maps.append(np.load(os.path.join(data_dir, f)))
        
        print(f"Loaded {len(self.lidar_data)} samples")
    
    def save_data(self, data_dir: str):
        """保存数据"""
        os.makedirs(data_dir, exist_ok=True)
        
        np.save(os.path.join(data_dir, 'lidar.npy'), self.lidar_data)
        np.save(os.path.join(data_dir, 'positions.npy'), self.positions)
        np.save(os.path.join(data_dir, 'headings.npy'), self.headings)
        np.save(os.path.join(data_dir, 'velocities.npy'), self.velocities)
        np.save(os.path.join(data_dir, 'accelerations.npy'), self.accelerations)
        np.save(os.path.join(data_dir, 'goals.npy'), self.goals)
        np.save(os.path.join(data_dir, 'map_indices.npy'), self.map_indices)
        
        for i, esdf in enumerate(self.esdf_maps):
            np.save(os.path.join(data_dir, f'esdf_{i}.npy'), esdf)
        
        print(f"Saved dataset to {data_dir}")
    
    def _split_data(self):
        """划分训练/验证集"""
        n_total = len(self.lidar_data)
        n_val = int(n_total * self.val_ratio)
        
        # 随机索引
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
            goal: [2] 世界目标
            map_idx: scalar 地图索引
        """
        i = self.indices[idx]
        
        if self.resample_state:
            vel = self._sample_velocity()
            acc = self._sample_acceleration()
            goal = self._sample_goal(self.positions[i], self.headings[i])
        else:
            vel = self.velocities[i]
            acc = self.accelerations[i]
            goal = self.goals[i]

        return (
            torch.tensor(self.lidar_data[i]),
            torch.tensor(self.positions[i]),
            torch.tensor(self.headings[i]),
            torch.tensor(vel),
            torch.tensor(acc),
            torch.tensor(goal),
            torch.tensor(self.map_indices[i])
        )
    
    def get_state_body(self, idx) -> torch.Tensor:
        """获取机体坐标系下的状态 [vx, vy, ax, ay, goal_x_body, goal_y_body]"""
        i = self.indices[idx]
        
        pos = self.positions[i]
        heading = self.headings[i]
        if self.resample_state:
            vel = self._sample_velocity()
            acc = self._sample_acceleration()
            goal = self._sample_goal(pos, heading)
        else:
            goal = self.goals[i]
            vel = self.velocities[i]
            acc = self.accelerations[i]
        
        # 目标转到机体系
        goal_rel = goal - pos
        c, s = np.cos(-heading), np.sin(-heading)
        goal_body = np.array([
            c * goal_rel[0] - s * goal_rel[1],
            s * goal_rel[0] + c * goal_rel[1]
        ])
        
        state = np.concatenate([vel, acc, goal_body])
        return torch.tensor(state, dtype=torch.float32)


def generate_dataset(
    save_dir: str = None,
    num_maps: int = None,
    samples_per_map: int = None
):
    """生成并保存数据集"""
    num_maps = num_maps or 10
    samples_per_map = samples_per_map or cfg['training']['num_samples'] // num_maps
    
    dataset = YopoDataset2D(
        data_dir=None,
        mode='train',
        num_maps=num_maps,
        samples_per_map=samples_per_map
    )
    
    if save_dir:
        dataset.save_data(save_dir)
    
    return dataset


if __name__ == "__main__":
    # 测试数据集
    print("Testing dataset generation...")
    
    train_dataset = YopoDataset2D(
        data_dir=None,
        mode='train',
        num_maps=10,
        samples_per_map=5000
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # 获取样本
    sample = train_dataset[0]
    print(f"Sample shapes: {[s.shape for s in sample]}")
    
    # 测试DataLoader
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    batch = next(iter(dataloader))
    print(f"Batch shapes: {[b.shape for b in batch]}")
    
    # 保存数据集
    train_dataset.save_data('./dataset/dataset2')
