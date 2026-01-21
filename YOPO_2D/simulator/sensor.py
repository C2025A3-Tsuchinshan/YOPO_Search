"""
2D传感器仿真器
实现2D激光雷达传感器
"""

import numpy as np
from typing import Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from simulator.map_generator import Map2D

# CUDA加速 (可选)
try:
    from simulator.cuda_accelerator import get_accelerator, CUDA_AVAILABLE
    USE_CUDA = CUDA_AVAILABLE
except ImportError:
    USE_CUDA = False


class Lidar2D:
    """2D激光雷达传感器 (支持CUDA加速)"""
    
    def __init__(
        self,
        num_beams: int = None,
        fov: float = None,
        max_range: float = None,
        min_range: float = None,
        noise_std: float = None,
        use_cuda: bool = None
    ):
        """
        Args:
            num_beams: 激光束数量
            fov: 视场角 (degrees)
            max_range: 最大探测距离 (meters)
            min_range: 最小探测距离 (meters)
            noise_std: 测量噪声标准差 (meters)
            use_cuda: 是否使用CUDA加速，None表示自动检测
        """
        self.num_beams = num_beams or cfg['sensor']['num_beams']
        self.fov = np.deg2rad(fov or cfg['sensor']['fov'])
        self.max_range = max_range or cfg['sensor']['max_range']
        self.min_range = min_range or cfg['sensor']['min_range']
        self.noise_std = noise_std or cfg['sensor']['noise_std']
        self.use_cuda = use_cuda if use_cuda is not None else USE_CUDA
        
        # 预计算光束角度 (以机器人朝向为0度, 逆时针为正)
        # 注意：当 fov=360° 时，如果使用 endpoint=True 会导致 -pi 与 +pi 重复（同一方向两次采样）。
        # 因此全向雷达使用 endpoint=False。
        full_circle = self.fov >= (2 * np.pi - 1e-6)
        if full_circle:
            self.beam_angles = np.linspace(-np.pi, np.pi, self.num_beams, endpoint=False)
        else:
            self.beam_angles = np.linspace(-self.fov / 2, self.fov / 2, self.num_beams, endpoint=True)
        
        # CUDA加速器
        self._accelerator = None
        if self.use_cuda:
            try:
                self._accelerator = get_accelerator()
            except Exception as e:
                print(f"[Warning] CUDA accelerator init failed: {e}")
                self.use_cuda = False
    
    def scan(
        self,
        map_2d: Map2D,
        position: np.ndarray,
        heading: float,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        执行激光扫描
        
        Args:
            map_2d: 2D地图
            position: 机器人位置 [x, y]
            heading: 机器人朝向 (radians)
            add_noise: 是否添加噪声
            
        Returns:
            ranges: 各光束测量距离 [num_beams]
        """
        position = np.asarray(position)
        
        # 全局光束角度
        global_angles = self.beam_angles + heading
        
        # 光束方向向量
        directions = np.stack([
            np.cos(global_angles),
            np.sin(global_angles)
        ], axis=1)  # [num_beams, 2]
        
        # 使用射线投射获取距离
        if self.use_cuda and self._accelerator is not None:
            try:
                ranges = self._raycast_cuda(map_2d, position, directions)
            except Exception as e:
                # 回退到CPU版本
                ranges = self._raycast_batch(map_2d, position, directions)
        else:
            ranges = self._raycast_batch(map_2d, position, directions)
        
        # 添加噪声
        if add_noise and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, self.num_beams)
            ranges = np.clip(ranges + noise, self.min_range, self.max_range)
        
        return ranges.astype(np.float32)
    
    def _raycast_cuda(
        self,
        map_2d: Map2D,
        origin: np.ndarray,
        directions: np.ndarray
    ) -> np.ndarray:
        """
        CUDA加速的射线投射
        
        Args:
            map_2d: 2D地图
            origin: 射线起点 [x, y]
            directions: 射线方向 [num_rays, 2]
            
        Returns:
            distances: 各射线击中距离 [num_rays]
        """
        return self._accelerator.raycast(
            map_2d.grid,
            origin,
            directions,
            self.max_range,
            self.min_range,
            map_2d.resolution
        )
    
    def _raycast_batch(
        self,
        map_2d: Map2D,
        origin: np.ndarray,
        directions: np.ndarray
    ) -> np.ndarray:
        """
        批量射线投射
        
        Args:
            map_2d: 2D地图
            origin: 射线起点 [x, y]
            directions: 射线方向 [num_rays, 2]
            
        Returns:
            distances: 各射线击中距离 [num_rays]
        """
        num_rays = len(directions)
        distances = np.full(num_rays, self.max_range)
        
        # 步进参数
        step_size = map_2d.resolution * 0.5
        max_steps = int(self.max_range / step_size)
        
        # 对每个射线进行步进
        for step in range(max_steps):
            t = self.min_range + step * step_size
            if t > self.max_range:
                break
            
            # 计算当前位置
            points = origin + t * directions  # [num_rays, 2]
            
            # 检查边界
            in_bounds = (
                (points[:, 0] >= 0) & (points[:, 0] < map_2d.size[0]) &
                (points[:, 1] >= 0) & (points[:, 1] < map_2d.size[1])
            )
            
            # 检查碰撞
            grid_pos = map_2d.world_to_grid(points)
            occupied = map_2d.grid[grid_pos[:, 0], grid_pos[:, 1]] > 0.5
            
            # 更新距离 (首次碰撞)
            hit = in_bounds & occupied & (distances == self.max_range)
            distances[hit] = t
            
            # 出界的射线也记录距离
            out_of_bounds = ~in_bounds & (distances == self.max_range)
            distances[out_of_bounds] = t
        
        return distances
    
    def scan_to_points(
        self,
        ranges: np.ndarray,
        position: np.ndarray,
        heading: float
    ) -> np.ndarray:
        """
        将扫描距离转换为点云
        
        Args:
            ranges: 测量距离 [num_beams]
            position: 机器人位置 [x, y]
            heading: 机器人朝向 (radians)
            
        Returns:
            points: 点云坐标 [num_beams, 2]
        """
        global_angles = self.beam_angles + heading
        
        points = position + ranges[:, np.newaxis] * np.stack([
            np.cos(global_angles),
            np.sin(global_angles)
        ], axis=1)
        
        return points
    
    def normalize_ranges(self, ranges: np.ndarray) -> np.ndarray:
        """归一化距离值到 [0, 1]"""
        return (ranges - self.min_range) / (self.max_range - self.min_range)
    
    def denormalize_ranges(self, normalized: np.ndarray) -> np.ndarray:
        """反归一化距离值"""
        return normalized * (self.max_range - self.min_range) + self.min_range


class SensorSimulator:
    """传感器仿真器（可扩展支持多种传感器）"""
    
    def __init__(self):
        self.lidar = Lidar2D()
    
    def get_observation(
        self,
        map_2d: Map2D,
        position: np.ndarray,
        heading: float,
        normalize: bool = True
    ) -> dict:
        """
        获取传感器观测
        
        Args:
            map_2d: 2D地图
            position: 机器人位置
            heading: 机器人朝向
            normalize: 是否归一化
            
        Returns:
            observation: {'lidar': ranges, 'points': point_cloud}
        """
        ranges = self.lidar.scan(map_2d, position, heading)
        points = self.lidar.scan_to_points(ranges, position, heading)
        
        if normalize:
            ranges_normalized = self.lidar.normalize_ranges(ranges)
        else:
            ranges_normalized = ranges
        
        return {
            'lidar': ranges_normalized,
            'lidar_raw': ranges,
            'points': points
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from map_generator import Map2D
    
    # 创建地图
    map_2d = Map2D(size=[30, 30], seed=42)
    map_2d.generate('forest', num_obstacles=30)
    
    # 创建传感器
    lidar = Lidar2D()
    
    # 采样有效位置
    pos = map_2d.sample_free_position()
    heading = np.random.uniform(-np.pi, np.pi)
    
    # 执行扫描
    ranges = lidar.scan(map_2d, pos, heading)
    points = lidar.scan_to_points(ranges, pos, heading)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 地图和点云
    ax = axes[0]
    ax.imshow(map_2d.grid.T, origin='lower', cmap='binary', 
              extent=[0, map_2d.size[0], 0, map_2d.size[1]])
    ax.plot(pos[0], pos[1], 'go', markersize=10, label='Robot')
    ax.quiver(pos[0], pos[1], np.cos(heading), np.sin(heading), 
              color='g', scale=5, width=0.02)
    ax.scatter(points[:, 0], points[:, 1], c='r', s=2, label='Lidar Points')
    ax.set_title('Map and Lidar Scan')
    ax.legend()
    ax.set_aspect('equal')
    
    # 距离曲线
    ax = axes[1]
    ax.plot(np.rad2deg(lidar.beam_angles), ranges)
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Range (m)')
    ax.set_title('Lidar Range Profile')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('lidar_test.png')
    plt.show()
