"""
2D环境生成器
生成各种类型的2D障碍物地图
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.ndimage import binary_dilation, distance_transform_edt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg

# CUDA加速 (可选)
try:
    from simulator.cuda_accelerator import get_accelerator, CUDA_AVAILABLE
    USE_CUDA = CUDA_AVAILABLE
except ImportError:
    USE_CUDA = False


class Map2D:
    """2D占用栅格地图"""
    
    def __init__(
        self,
        size: Tuple[float, float] = None,
        resolution: float = None,
        seed: int = None
    ):
        """
        Args:
            size: 地图大小 [x, y] (meters)
            resolution: 分辨率 (meters/grid)
            seed: 随机种子
        """
        self.size = size or cfg['env']['map_size']
        self.resolution = resolution or cfg['env']['resolution']
        self.seed = seed or cfg['env']['seed']
        
        # 计算栅格尺寸
        self.grid_size = (
            int(self.size[0] / self.resolution),
            int(self.size[1] / self.resolution)
        )
        
        # 占用栅格 (0: free, 1: occupied)
        self.grid = np.zeros(self.grid_size, dtype=np.float32)
        
        # 距离场 (ESDF)
        self.esdf = None
        
        # 障碍物列表 (用于解析碰撞检测)
        self.obstacles: List[dict] = []
        
        # 随机数生成器
        self.rng = np.random.default_rng(self.seed)
    
    def world_to_grid(self, pos: np.ndarray) -> np.ndarray:
        """世界坐标转栅格坐标"""
        pos = np.asarray(pos)
        grid_pos = (pos / self.resolution).astype(int)
        return np.clip(grid_pos, [0, 0], [self.grid_size[0]-1, self.grid_size[1]-1])
    
    def grid_to_world(self, grid_pos: np.ndarray) -> np.ndarray:
        """栅格坐标转世界坐标"""
        return (np.asarray(grid_pos) + 0.5) * self.resolution
    
    def is_occupied(self, pos: np.ndarray) -> bool:
        """检查位置是否被占用"""
        grid_pos = self.world_to_grid(pos)
        if grid_pos.ndim == 1:
            return self.grid[grid_pos[0], grid_pos[1]] > 0.5
        return self.grid[grid_pos[:, 0], grid_pos[:, 1]] > 0.5
    
    def is_valid_position(self, pos: np.ndarray, radius: float = 0.0) -> bool:
        """检查位置是否有效（在地图内且未碰撞）"""
        pos = np.asarray(pos)
        # 检查边界
        if np.any(pos < radius) or np.any(pos > np.array(self.size) - radius):
            return False
        # 检查碰撞
        if radius > 0 and self.esdf is not None:
            grid_pos = self.world_to_grid(pos)
            return self.esdf[grid_pos[0], grid_pos[1]] * self.resolution > radius
        return not self.is_occupied(pos)
    
    def get_distance(self, pos: np.ndarray) -> float:
        """获取位置到最近障碍物的距离"""
        if self.esdf is None:
            self.compute_esdf()
        grid_pos = self.world_to_grid(pos)
        if grid_pos.ndim == 1:
            return self.esdf[grid_pos[0], grid_pos[1]] * self.resolution
        return self.esdf[grid_pos[:, 0], grid_pos[:, 1]] * self.resolution
    
    def compute_esdf(self, use_cuda: bool = None):
        """计算欧几里得符号距离场
        
        Args:
            use_cuda: 是否使用CUDA加速，None表示自动检测
        """
        use_cuda = use_cuda if use_cuda is not None else USE_CUDA
        
        if use_cuda:
            try:
                accelerator = get_accelerator()
                self.esdf = accelerator.compute_esdf(self.grid)
            except Exception as e:
                print(f"[Warning] CUDA ESDF failed, falling back to CPU: {e}")
                self.esdf = distance_transform_edt(1 - self.grid)
        else:
            # CPU版本 (scipy)
            self.esdf = distance_transform_edt(1 - self.grid)
    
    def add_boundary(self, thickness: float = 0.5):
        """添加边界墙"""
        t = int(thickness / self.resolution)
        self.grid[:t, :] = 1.0
        self.grid[-t:, :] = 1.0
        self.grid[:, :t] = 1.0
        self.grid[:, -t:] = 1.0
    
    def add_circle(self, center: Tuple[float, float], radius: float):
        """添加圆形障碍物"""
        cx, cy = int(center[0] / self.resolution), int(center[1] / self.resolution)
        r = int(radius / self.resolution)
        
        y, x = np.ogrid[-cx:self.grid_size[0]-cx, -cy:self.grid_size[1]-cy]
        mask = x*x + y*y <= r*r
        self.grid[mask] = 1.0
        
        self.obstacles.append({
            'type': 'circle',
            'center': center,
            'radius': radius
        })
    
    def add_rectangle(self, center: Tuple[float, float], size: Tuple[float, float], angle: float = 0):
        """添加矩形障碍物"""
        # 简化实现: 不考虑旋转
        cx, cy = center
        w, h = size
        
        x_min = max(0, int((cx - w/2) / self.resolution))
        x_max = min(self.grid_size[0], int((cx + w/2) / self.resolution))
        y_min = max(0, int((cy - h/2) / self.resolution))
        y_max = min(self.grid_size[1], int((cy + h/2) / self.resolution))
        
        self.grid[x_min:x_max, y_min:y_max] = 1.0
        
        self.obstacles.append({
            'type': 'rectangle',
            'center': center,
            'size': size,
            'angle': angle
        })
    
    def generate_forest(
        self,
        num_obstacles: int = None,
        radius_range: Tuple[float, float] = None
    ):
        """生成森林地图（随机圆形障碍物）"""
        num_obstacles = num_obstacles or cfg['env']['obstacle_num']
        radius_range = radius_range or (
            cfg['env']['obstacle_radius_min'],
            cfg['env']['obstacle_radius_max']
        )
        
        self.grid.fill(0)
        self.obstacles.clear()
        self.add_boundary()
        
        for _ in range(num_obstacles):
            radius = self.rng.uniform(*radius_range)
            # 随机位置（避开边界）
            margin = radius + 1.0
            cx = self.rng.uniform(margin, self.size[0] - margin)
            cy = self.rng.uniform(margin, self.size[1] - margin)
            self.add_circle((cx, cy), radius)
        
        self.compute_esdf()
    
    def generate_maze(self, complexity: float = 0.3, density: float = 0.3):
        """生成迷宫地图"""
        self.grid.fill(0)
        self.obstacles.clear()
        
        
        shape = ((self.grid_size[0] // 4) * 2 + 1, (self.grid_size[1] // 4) * 2 + 1)
        
        # 调整复杂度和密度
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        density = int(density * ((shape[0] // 2) * (shape[1] // 2)))
        
        maze = np.zeros(shape, dtype=bool)
        
        # 填充边界
        maze[0, :] = maze[-1, :] = True
        maze[:, 0] = maze[:, -1] = True
        
        
        # 生成迷宫
        for _ in range(density):
            x = self.rng.integers(0, shape[0] // 2 + 1) * 2
            y = self.rng.integers(0, shape[1] // 2 + 1) * 2
            maze[x, y] = True
            
            for _ in range(complexity):
                neighbours = []
                if x > 1: neighbours.append((x - 2, y))
                if x < shape[0] - 2: neighbours.append((x + 2, y))
                if y > 1: neighbours.append((x, y - 2))
                if y < shape[1] - 2: neighbours.append((x, y + 2))
                
                if len(neighbours):
                    x_, y_ = neighbours[self.rng.integers(0, len(neighbours))]
                    if not maze[x_, y_]:
                        maze[x_, y_] = True
                        maze[x_ + (x - x_) // 2, y_ + (y - y_) // 2] = True
                        x, y = x_, y_
        
        
        # 放大到目标尺寸
        scale_x = self.grid_size[0] // shape[0]
        scale_y = self.grid_size[1] // shape[1]
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                if maze[i, j]:
                    self.grid[i*scale_x:(i+1)*scale_x, j*scale_y:(j+1)*scale_y] = 1.0
        
        self.compute_esdf()
    
    def generate_pillars(
        self,
        num_pillars: int = None,
        radius_range: Tuple[float, float] = None,
        grid_spacing: float = None
    ):
        """生成柱子地图（规则排列的圆形障碍物）"""
        num_pillars = num_pillars or cfg['env']['obstacle_num']
        radius_range = radius_range or (
            cfg['env']['obstacle_radius_min'],
            cfg['env']['obstacle_radius_max']
        )
        
        self.grid.fill(0)
        self.obstacles.clear()
        self.add_boundary()
        
        # 计算网格
        n_x = int(np.sqrt(num_pillars * self.size[0] / self.size[1]))
        n_y = int(num_pillars / n_x)
        
        spacing_x = self.size[0] / (n_x + 1)
        spacing_y = self.size[1] / (n_y + 1)
        
        for i in range(n_x):
            for j in range(n_y):
                # 添加随机偏移
                offset_x = self.rng.uniform(-spacing_x * 0.3, spacing_x * 0.3)
                offset_y = self.rng.uniform(-spacing_y * 0.3, spacing_y * 0.3)
                
                cx = (i + 1) * spacing_x + offset_x
                cy = (j + 1) * spacing_y + offset_y
                
                radius = self.rng.uniform(*radius_range)
                
                # 确保在边界内
                cx = np.clip(cx, radius + 0.5, self.size[0] - radius - 0.5)
                cy = np.clip(cy, radius + 0.5, self.size[1] - radius - 0.5)
                
                self.add_circle((cx, cy), radius)
        
        self.compute_esdf()
    
    def generate(self, map_type: str = None, **kwargs):
        """根据类型生成地图"""
        map_type = map_type or cfg['env']['map_type']
        
        if map_type == 'forest':
            self.generate_forest(**kwargs)
        elif map_type == 'maze':
            self.generate_maze(**kwargs)
        elif map_type == 'pillars':
            self.generate_pillars(**kwargs)
        else:
            # 默认森林
            self.generate_forest(**kwargs)
        
        return self
    
    def sample_free_position(self, robot_radius: float = None, max_attempts: int = 1000) -> Optional[np.ndarray]:
        """在自由空间中采样有效位置"""
        robot_radius = robot_radius or cfg['robot']['radius']
        
        for _ in range(max_attempts):
            pos = self.rng.uniform([robot_radius, robot_radius], 
                                   [self.size[0] - robot_radius, self.size[1] - robot_radius])
            if self.is_valid_position(pos, robot_radius):
                return pos
        return None
    
    def get_esdf_batch(self, positions: np.ndarray) -> np.ndarray:
        """批量获取ESDF值"""
        if self.esdf is None:
            self.compute_esdf()
        
        grid_pos = self.world_to_grid(positions)
        return self.esdf[grid_pos[:, 0], grid_pos[:, 1]] * self.resolution


class MapManager:
    """地图管理器，用于训练时管理多个地图"""
    
    def __init__(self, num_maps: int = 10, seed: int = None):
        self.num_maps = num_maps
        self.seed = seed or cfg['env']['seed']
        self.maps: List[Map2D] = []
        
        self._generate_maps()
    
    def _generate_maps(self):
        """生成多个地图"""
        print(f"Generating {self.num_maps} maps...")
        for i in range(self.num_maps):
            map_2d = Map2D(seed=self.seed + i)
            map_2d.generate()
            self.maps.append(map_2d)
        print("Maps generated!")
    
    def get_map(self, idx: int) -> Map2D:
        """获取指定索引的地图"""
        return self.maps[idx % self.num_maps]
    
    def get_random_map(self) -> Tuple[int, Map2D]:
        """随机获取地图"""
        idx = np.random.randint(0, self.num_maps)
        return idx, self.maps[idx]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 测试地图生成
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    map_types = ['forest', 'maze', 'pillars']
    for ax, map_type in zip(axes, map_types):
        map_2d = Map2D(seed=42)
        map_2d.generate(map_type)
        
        ax.imshow(map_2d.grid.T, origin='lower', cmap='binary')
        ax.set_title(f'{map_type.capitalize()} Map')
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Y (grid)')
    
    plt.tight_layout()
    plt.savefig('maps_test.png')
    plt.show()
