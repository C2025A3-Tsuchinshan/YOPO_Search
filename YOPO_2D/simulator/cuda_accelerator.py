"""
CUDA加速模块
实现ESDF计算、射线投射、碰撞检测的GPU加速版本

支持两种后端:
1. PyTorch (推荐，更好的兼容性)
2. CuPy (可选，某些操作更快)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import warnings

# 检测CUDA可用性
CUDA_AVAILABLE = torch.cuda.is_available()
CUPY_AVAILABLE = False

try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cupy_distance_transform
    CUPY_AVAILABLE = cp.cuda.is_available()
except (ImportError, AttributeError, Exception) as e:
    # CuPy可能因为NumPy版本不兼容而导入失败
    CUPY_AVAILABLE = False
    pass

if CUDA_AVAILABLE:
    print(f"[CUDA] PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
if CUPY_AVAILABLE:
    print(f"[CUDA] CuPy available")


class CUDAAccelerator:
    """CUDA加速器单例类"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        self.use_cupy = CUPY_AVAILABLE
        
        # 预编译的CUDA kernel缓存
        self._kernels = {}
        
        print(f"[CUDAAccelerator] Device: {self.device}, CuPy: {self.use_cupy}")
    
    def to_torch(self, arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        """NumPy转PyTorch Tensor (GPU)"""
        return torch.tensor(arr, dtype=dtype, device=self.device)
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """PyTorch Tensor转NumPy"""
        return tensor.cpu().numpy()


# ============================================================================
# ESDF 计算 (GPU加速)
# ============================================================================

class ESDFCUDA:
    """
    GPU加速的欧几里得符号距离场计算
    
    使用Jump Flooding Algorithm (JFA) 实现近似ESDF
    JFA时间复杂度: O(N * log(max_dim))，比CPU的O(N * max_dim)快很多
    
    对于小地图(<2000x2000)，scipy的CPU版本可能更快，建议使用auto模式
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    
    def compute_esdf_torch(self, grid: np.ndarray) -> np.ndarray:
        """
        使用PyTorch计算ESDF (优化的Jump Flooding Algorithm)
        
        Args:
            grid: [H, W] 占用栅格 (0: free, 1: occupied)
            
        Returns:
            esdf: [H, W] 距离场 (单位: 栅格)
        """
        H, W = grid.shape
        
        # 对于小地图，直接使用scipy（更快）
        if H * W < 500 * 500:
            from scipy.ndimage import distance_transform_edt
            return distance_transform_edt(1 - grid)
        
        # 转换到GPU
        grid_t = torch.tensor(grid, dtype=torch.float32, device=self.device)
        obstacle_mask = grid_t > 0.5
        
        # 创建坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # 初始化: 障碍物点的最近点是自己，自由空间初始化为无穷远
        INF = float(H + W) * 2
        nearest_x = torch.where(obstacle_mask, x_coords, torch.full_like(x_coords, INF))
        nearest_y = torch.where(obstacle_mask, y_coords, torch.full_like(y_coords, INF))
        
        # 预计算8个方向的偏移索引
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # Jump Flooding Algorithm (优化版：使用torch.roll代替切片)
        max_dim = max(H, W)
        step = 1
        while step < max_dim:
            step *= 2
        step //= 2
        
        while step >= 1:
            # 批量处理所有8个方向
            for dy, dx in offsets:
                shift_y = dy * step
                shift_x = dx * step
                
                # 使用roll进行移位（更快）
                neighbor_x = torch.roll(nearest_x, shifts=(-shift_y, -shift_x), dims=(0, 1))
                neighbor_y = torch.roll(nearest_y, shifts=(-shift_y, -shift_x), dims=(0, 1))
                
                # 处理边界：将roll过来的无效区域设为INF
                if shift_y > 0:
                    neighbor_x[-shift_y:, :] = INF
                    neighbor_y[-shift_y:, :] = INF
                elif shift_y < 0:
                    neighbor_x[:-shift_y, :] = INF
                    neighbor_y[:-shift_y, :] = INF
                    
                if shift_x > 0:
                    neighbor_x[:, -shift_x:] = INF
                    neighbor_y[:, -shift_x:] = INF
                elif shift_x < 0:
                    neighbor_x[:, :-shift_x] = INF
                    neighbor_y[:, :-shift_x] = INF
                
                # 计算距离
                current_dist_sq = (x_coords - nearest_x) ** 2 + (y_coords - nearest_y) ** 2
                neighbor_dist_sq = (x_coords - neighbor_x) ** 2 + (y_coords - neighbor_y) ** 2
                
                # 更新最近点
                update_mask = neighbor_dist_sq < current_dist_sq
                nearest_x = torch.where(update_mask, neighbor_x, nearest_x)
                nearest_y = torch.where(update_mask, neighbor_y, nearest_y)
            
            step //= 2
        
        # 计算最终距离
        esdf = torch.sqrt((x_coords - nearest_x) ** 2 + (y_coords - nearest_y) ** 2)
        
        return esdf.cpu().numpy()
    
    def _shift_2d(self, tensor: torch.Tensor, dy: int, dx: int, fill_value: float) -> torch.Tensor:
        """2D张量移位 (备用方法)"""
        H, W = tensor.shape
        result = torch.full_like(tensor, fill_value)
        
        # 计算有效区域
        src_y_start = max(0, -dy)
        src_y_end = min(H, H - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(W, W - dx)
        
        dst_y_start = max(0, dy)
        dst_y_end = min(H, H + dy)
        dst_x_start = max(0, dx)
        dst_x_end = min(W, W + dx)
        
        if src_y_end > src_y_start and src_x_end > src_x_start:
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                tensor[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return result
    
    def compute_esdf_cupy(self, grid: np.ndarray) -> np.ndarray:
        """
        使用CuPy的scipy接口计算精确ESDF
        
        Args:
            grid: [H, W] 占用栅格
            
        Returns:
            esdf: [H, W] 距离场
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        
        grid_gpu = cp.asarray(grid)
        free_space = 1 - grid_gpu
        esdf_gpu = cupy_distance_transform(free_space)
        return cp.asnumpy(esdf_gpu)
    
    def compute_esdf(self, grid: np.ndarray, method: str = 'auto') -> np.ndarray:
        """
        计算ESDF的统一接口
        
        Args:
            grid: [H, W] 占用栅格
            method: 'scipy', 'torch', 'cupy', 'auto'
            
        Returns:
            esdf: [H, W] 距离场
            
        Note:
            scipy的EDT实现高度优化，通常比GPU实现更快。
            仅在需要保持GPU张量流水线时才使用torch/cupy方法。
        """
        if method == 'auto':
            # scipy通常最快，作为默认选项
            from scipy.ndimage import distance_transform_edt
            return distance_transform_edt(1 - grid)
        
        if method == 'cupy':
            if CUPY_AVAILABLE:
                return self.compute_esdf_cupy(grid)
            else:
                from scipy.ndimage import distance_transform_edt
                return distance_transform_edt(1 - grid)
        elif method == 'torch':
            if CUDA_AVAILABLE:
                return self.compute_esdf_torch(grid)
            else:
                from scipy.ndimage import distance_transform_edt
                return distance_transform_edt(1 - grid)
        else:
            from scipy.ndimage import distance_transform_edt
            return distance_transform_edt(1 - grid)
    
    def compute_esdf_gpu_tensor(self, grid: torch.Tensor) -> torch.Tensor:
        """
        在GPU上计算ESDF并返回GPU张量（用于避免CPU-GPU数据传输）
        
        Args:
            grid: [H, W] 占用栅格 (GPU tensor)
            
        Returns:
            esdf: [H, W] 距离场 (GPU tensor)
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
        
        # 直接使用GPU张量进行计算
        H, W = grid.shape
        max_dim = max(H, W)
        
        # 自由空间
        free_space = 1 - grid
        
        # 初始化距离场
        esdf = torch.where(
            free_space > 0.5,
            torch.zeros_like(grid),
            torch.full_like(grid, float('inf'))
        )
        
        # JFA迭代
        num_iterations = int(np.ceil(np.log2(max_dim)))
        
        for i in range(num_iterations):
            step = 2 ** (num_iterations - i - 1)
            
            # 水平和垂直偏移
            for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                          (step, step), (step, -step), (-step, step), (-step, -step)]:
                shifted = torch.roll(torch.roll(esdf, dx, dims=0), dy, dims=1)
                new_dist = shifted + np.sqrt(dx**2 + dy**2)
                esdf = torch.minimum(esdf, new_dist)
        
        return esdf


# ============================================================================
# 射线投射 (GPU加速)
# ============================================================================

class RaycastCUDA:
    """
    GPU加速的射线投射
    
    使用并行化的DDA算法或步进法
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    
    def raycast_batch_torch(
        self,
        grid: torch.Tensor,
        origin: torch.Tensor,
        directions: torch.Tensor,
        max_range: float,
        min_range: float = 0.1,
        resolution: float = 0.1
    ) -> torch.Tensor:
        """
        GPU并行射线投射 (步进法)
        
        Args:
            grid: [H, W] 占用栅格 (GPU tensor)
            origin: [2] 射线起点 (世界坐标)
            directions: [N, 2] 射线方向 (单位向量)
            max_range: 最大距离
            min_range: 最小距离
            resolution: 地图分辨率
            
        Returns:
            distances: [N] 各射线命中距离
        """
        N = directions.shape[0]
        H, W = grid.shape
        
        # 步进参数
        step_size = resolution * 0.5
        num_steps = int((max_range - min_range) / step_size) + 1
        
        # 生成采样距离 [num_steps]
        t_values = torch.linspace(min_range, max_range, num_steps, device=self.device)
        
        # 计算所有采样点 [N, num_steps, 2]
        # origin: [2] -> [1, 1, 2]
        # directions: [N, 2] -> [N, 1, 2]
        # t_values: [num_steps] -> [1, num_steps, 1]
        points = origin.view(1, 1, 2) + directions.unsqueeze(1) * t_values.view(1, -1, 1)
        
        # 转换到栅格坐标
        grid_coords = (points / resolution).long()
        grid_x = grid_coords[..., 0].clamp(0, H - 1)
        grid_y = grid_coords[..., 1].clamp(0, W - 1)
        
        # 检查边界
        in_bounds = (
            (points[..., 0] >= 0) & (points[..., 0] < H * resolution) &
            (points[..., 1] >= 0) & (points[..., 1] < W * resolution)
        )
        
        # 查询占用状态 [N, num_steps]
        occupied = grid[grid_x, grid_y] > 0.5
        
        # 结合边界检查: 出界也视为命中
        hit = occupied | (~in_bounds)
        
        # 找到每条射线的第一个命中点
        # 使用argmax找到第一个True的位置
        hit_float = hit.float()
        hit_float[~hit] = float('inf')
        
        # 累积求和找第一个命中
        cumsum = torch.cumsum(hit.float(), dim=1)
        first_hit_mask = (cumsum == 1) & hit
        
        # 获取第一个命中的索引
        first_hit_idx = first_hit_mask.float().argmax(dim=1)
        
        # 获取对应的距离
        distances = t_values[first_hit_idx]
        
        # 如果没有命中，返回max_range
        no_hit = ~hit.any(dim=1)
        distances[no_hit] = max_range
        
        return distances
    
    def raycast_batch_optimized(
        self,
        grid: torch.Tensor,
        origin: torch.Tensor,
        directions: torch.Tensor,
        max_range: float,
        min_range: float = 0.1,
        resolution: float = 0.1,
        batch_steps: int = 32
    ) -> torch.Tensor:
        """
        优化的GPU射线投射 (分批步进，减少内存使用)
        
        Args:
            grid: [H, W] 占用栅格
            origin: [2] 射线起点
            directions: [N, 2] 射线方向
            max_range, min_range: 距离范围
            resolution: 地图分辨率
            batch_steps: 每批处理的步数
            
        Returns:
            distances: [N] 各射线命中距离
        """
        N = directions.shape[0]
        H, W = grid.shape
        
        step_size = resolution * 0.5
        
        # 初始化距离为max_range
        distances = torch.full((N,), max_range, device=self.device, dtype=torch.float32)
        active = torch.ones(N, dtype=torch.bool, device=self.device)
        
        t = min_range
        while t < max_range and active.any():
            # 只处理活跃的射线
            active_idx = torch.where(active)[0]
            active_dirs = directions[active_idx]
            
            # 当前批次的采样距离
            batch_t = torch.arange(
                t, min(t + batch_steps * step_size, max_range), 
                step_size, device=self.device
            )
            
            if len(batch_t) == 0:
                break
            
            # 计算采样点 [active_N, batch_steps, 2]
            points = origin.view(1, 1, 2) + active_dirs.unsqueeze(1) * batch_t.view(1, -1, 1)
            
            # 转换到栅格坐标
            grid_x = (points[..., 0] / resolution).long().clamp(0, H - 1)
            grid_y = (points[..., 1] / resolution).long().clamp(0, W - 1)
            
            # 边界检查
            in_bounds = (
                (points[..., 0] >= 0) & (points[..., 0] < H * resolution) &
                (points[..., 1] >= 0) & (points[..., 1] < W * resolution)
            )
            
            # 查询占用
            occupied = grid[grid_x, grid_y] > 0.5
            hit = occupied | (~in_bounds)
            
            # 找每条射线的第一个命中
            hit_any = hit.any(dim=1)
            if hit_any.any():
                # 找命中的射线和对应的步数
                hit_rays = torch.where(hit_any)[0]
                for ray_local_idx in hit_rays:
                    ray_global_idx = active_idx[ray_local_idx]
                    hit_steps = torch.where(hit[ray_local_idx])[0]
                    if len(hit_steps) > 0:
                        first_hit_step = hit_steps[0]
                        distances[ray_global_idx] = batch_t[first_hit_step]
                        active[ray_global_idx] = False
            
            t += batch_steps * step_size
        
        return distances


# ============================================================================
# 碰撞检测 (GPU加速)
# ============================================================================

class CollisionCheckerCUDA:
    """
    GPU加速的碰撞检测
    
    基于ESDF的快速碰撞查询
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    
    def check_trajectory_collision(
        self,
        esdf: torch.Tensor,
        positions: torch.Tensor,
        robot_radius: float,
        resolution: float
    ) -> torch.Tensor:
        """
        检查轨迹点是否发生碰撞
        
        Args:
            esdf: [H, W] ESDF地图 (GPU tensor)
            positions: [B, N, 2] 轨迹点位置 (世界坐标)
            robot_radius: 机器人半径
            resolution: 地图分辨率
            
        Returns:
            collision: [B, N] 是否碰撞 (True=碰撞)
        """
        B, N, _ = positions.shape
        H, W = esdf.shape
        
        # 转换到栅格坐标
        grid_x = (positions[..., 0] / resolution).long().clamp(0, H - 1)
        grid_y = (positions[..., 1] / resolution).long().clamp(0, W - 1)
        
        # 边界检查
        out_of_bounds = (
            (positions[..., 0] < robot_radius) |
            (positions[..., 0] > H * resolution - robot_radius) |
            (positions[..., 1] < robot_radius) |
            (positions[..., 1] > W * resolution - robot_radius)
        )
        
        # 查询ESDF值
        distances = esdf[grid_x, grid_y] * resolution
        
        # 碰撞判断
        collision = (distances < robot_radius) | out_of_bounds
        
        return collision
    
    def check_trajectory_collision_bilinear(
        self,
        esdf: torch.Tensor,
        positions: torch.Tensor,
        robot_radius: float,
        resolution: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用双线性插值的碰撞检测 (更精确)
        
        Args:
            esdf: [H, W] ESDF地图
            positions: [B, N, 2] 轨迹点位置
            robot_radius: 机器人半径
            resolution: 地图分辨率
            
        Returns:
            collision: [B, N] 是否碰撞
            distances: [B, N] 到障碍物的距离
        """
        B, N, _ = positions.shape
        H, W = esdf.shape
        
        # 归一化到[-1, 1] (grid_sample要求)
        grid_x = (positions[..., 0] / resolution) / (H - 1) * 2 - 1
        grid_y = (positions[..., 1] / resolution) / (W - 1) * 2 - 1
        
        # 组合成grid_sample需要的格式 [B, N, 1, 2]
        grid = torch.stack([grid_y, grid_x], dim=-1).unsqueeze(2)
        
        # 使用grid_sample进行双线性插值
        esdf_4d = esdf.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        sampled = F.grid_sample(
            esdf_4d, grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        # [B, 1, N, 1] -> [B, N]
        distances = sampled.squeeze(1).squeeze(-1) * resolution
        
        # 边界检查
        out_of_bounds = (
            (positions[..., 0] < robot_radius) |
            (positions[..., 0] > H * resolution - robot_radius) |
            (positions[..., 1] < robot_radius) |
            (positions[..., 1] > W * resolution - robot_radius)
        )
        
        collision = (distances < robot_radius) | out_of_bounds
        
        return collision, distances
    
    def check_batch_trajectories(
        self,
        esdf_maps: torch.Tensor,
        positions: torch.Tensor,
        map_indices: torch.Tensor,
        robot_radius: float,
        resolution: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量检查多条轨迹在不同地图上的碰撞
        
        Args:
            esdf_maps: [M, H, W] 多个ESDF地图
            positions: [B, N, 2] 轨迹点位置
            map_indices: [B] 每条轨迹对应的地图索引
            robot_radius: 机器人半径
            resolution: 地图分辨率
            
        Returns:
            collision: [B, N] 是否碰撞
            distances: [B, N] 到障碍物的距离
        """
        B, N, _ = positions.shape
        M, H, W = esdf_maps.shape
        
        # 确保索引有效
        map_indices = map_indices % M
        
        # 转换到栅格坐标
        grid_x = (positions[..., 0] / resolution).clamp(0, H - 1)
        grid_y = (positions[..., 1] / resolution).clamp(0, W - 1)
        
        # 双线性插值的四个角点
        x0 = torch.floor(grid_x).long()
        y0 = torch.floor(grid_y).long()
        x1 = (x0 + 1).clamp(max=H - 1)
        y1 = (y0 + 1).clamp(max=W - 1)
        
        wx = grid_x - x0.float()
        wy = grid_y - y0.float()
        
        # 展平地图用于索引
        maps_flat = esdf_maps.view(M, -1)  # [M, H*W]
        map_idx_exp = map_indices.view(-1, 1).expand(-1, N)
        
        # 计算线性索引
        idx00 = x0 * W + y0
        idx10 = x1 * W + y0
        idx01 = x0 * W + y1
        idx11 = x1 * W + y1
        
        # 双线性插值
        v00 = maps_flat[map_idx_exp, idx00]
        v10 = maps_flat[map_idx_exp, idx10]
        v01 = maps_flat[map_idx_exp, idx01]
        v11 = maps_flat[map_idx_exp, idx11]
        
        distances = (
            (1 - wx) * (1 - wy) * v00 +
            wx * (1 - wy) * v10 +
            (1 - wx) * wy * v01 +
            wx * wy * v11
        ) * resolution
        
        # 边界检查
        out_of_bounds = (
            (positions[..., 0] < robot_radius) |
            (positions[..., 0] > H * resolution - robot_radius) |
            (positions[..., 1] < robot_radius) |
            (positions[..., 1] > W * resolution - robot_radius)
        )
        
        collision = (distances < robot_radius) | out_of_bounds
        
        return collision, distances


# ============================================================================
# 批量轨迹生成 (GPU加速)
# ============================================================================

class TrajectoryGeneratorCUDA:
    """
    GPU加速的五次多项式轨迹生成和采样
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    
    def generate_poly5_batch(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        T: float,
        num_samples: int = 20
    ) -> torch.Tensor:
        """
        批量生成五次多项式轨迹采样点
        
        Args:
            start_state: [B, 3, 2] = [[px, py], [vx, vy], [ax, ay]]
            end_state: [B, 3, 2]
            T: 轨迹时间
            num_samples: 采样点数
            
        Returns:
            positions: [B, num_samples, 2] 轨迹位置采样
        """
        B = start_state.shape[0]
        
        pos0 = start_state[:, 0, :]  # [B, 2]
        vel0 = start_state[:, 1, :]
        acc0 = start_state[:, 2, :]
        pos1 = end_state[:, 0, :]
        vel1 = end_state[:, 1, :]
        acc1 = end_state[:, 2, :]
        
        # 计算系数
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T
        
        a0 = pos0
        a1 = vel0
        a2 = acc0 * 0.5
        
        b0 = pos1 - (pos0 + vel0 * T + 0.5 * acc0 * T2)
        b1 = vel1 - (vel0 + acc0 * T)
        b2 = acc1 - acc0
        
        # 求解系数矩阵
        A = torch.tensor([
            [T3, T4, T5],
            [3 * T2, 4 * T3, 5 * T4],
            [6 * T, 12 * T2, 20 * T3]
        ], dtype=start_state.dtype, device=self.device)
        
        A_inv = torch.inverse(A)
        
        b = torch.stack([b0, b1, b2], dim=1)  # [B, 3, 2]
        a345 = torch.matmul(A_inv, b)  # [B, 3, 2]
        a3, a4, a5 = a345[:, 0, :], a345[:, 1, :], a345[:, 2, :]
        
        # 采样
        t = torch.linspace(0, T, num_samples, device=self.device)
        t1 = t.view(1, -1, 1)  # [1, num_samples, 1]
        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t3 * t1
        t5 = t4 * t1
        
        positions = (
            a0.unsqueeze(1) +
            a1.unsqueeze(1) * t1 +
            a2.unsqueeze(1) * t2 +
            a3.unsqueeze(1) * t3 +
            a4.unsqueeze(1) * t4 +
            a5.unsqueeze(1) * t5
        )
        
        return positions


# ============================================================================
# 综合加速器接口
# ============================================================================

class YOPO2DCUDAAccelerator:
    """
    YOPO 2D的综合CUDA加速接口
    
    整合ESDF计算、射线投射、碰撞检测
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        self.esdf_cuda = ESDFCUDA(self.device)
        self.raycast_cuda = RaycastCUDA(self.device)
        self.collision_cuda = CollisionCheckerCUDA(self.device)
        self.traj_cuda = TrajectoryGeneratorCUDA(self.device)
        
        # 缓存
        self._grid_cache = None
        self._esdf_cache = None
    
    @property
    def is_cuda_available(self) -> bool:
        return CUDA_AVAILABLE
    
    def compute_esdf(self, grid: np.ndarray) -> np.ndarray:
        """计算ESDF"""
        return self.esdf_cuda.compute_esdf(grid)
    
    def raycast(
        self,
        grid: np.ndarray,
        origin: np.ndarray,
        directions: np.ndarray,
        max_range: float,
        min_range: float = 0.1,
        resolution: float = 0.1
    ) -> np.ndarray:
        """射线投射"""
        grid_t = torch.tensor(grid, dtype=torch.float32, device=self.device)
        origin_t = torch.tensor(origin, dtype=torch.float32, device=self.device)
        dirs_t = torch.tensor(directions, dtype=torch.float32, device=self.device)
        
        # 使用全并行方法（避免Python循环）
        distances = self.raycast_cuda.raycast_batch_torch(
            grid_t, origin_t, dirs_t, max_range, min_range, resolution
        )
        
        return distances.cpu().numpy()
    
    def check_collision(
        self,
        esdf: np.ndarray,
        positions: np.ndarray,
        robot_radius: float,
        resolution: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """碰撞检测"""
        esdf_t = torch.tensor(esdf, dtype=torch.float32, device=self.device)
        
        if positions.ndim == 2:
            positions = positions[np.newaxis, ...]
        
        pos_t = torch.tensor(positions, dtype=torch.float32, device=self.device)
        
        collision, distances = self.collision_cuda.check_trajectory_collision_bilinear(
            esdf_t, pos_t, robot_radius, resolution
        )
        
        return collision.cpu().numpy(), distances.cpu().numpy()
    
    def generate_trajectories(
        self,
        start_state: np.ndarray,
        end_state: np.ndarray,
        T: float,
        num_samples: int = 20
    ) -> np.ndarray:
        """生成轨迹采样点"""
        start_t = torch.tensor(start_state, dtype=torch.float32, device=self.device)
        end_t = torch.tensor(end_state, dtype=torch.float32, device=self.device)
        
        positions = self.traj_cuda.generate_poly5_batch(start_t, end_t, T, num_samples)
        
        return positions.cpu().numpy()


# 全局实例
_accelerator = None

def get_accelerator() -> YOPO2DCUDAAccelerator:
    """获取全局加速器实例"""
    global _accelerator
    if _accelerator is None:
        _accelerator = YOPO2DCUDAAccelerator()
    return _accelerator


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("CUDA Accelerator Test")
    print("=" * 60)
    
    # 创建测试数据
    H, W = 1000, 1000
    grid = np.zeros((H, W), dtype=np.float32)
    
    # 添加一些障碍物
    for _ in range(50):
        cx, cy = np.random.randint(50, 950, 2)
        r = np.random.randint(10, 50)
        y, x = np.ogrid[-cx:H-cx, -cy:W-cy]
        mask = x*x + y*y <= r*r
        grid[mask] = 1.0
    
    accelerator = get_accelerator()
    
    # 测试ESDF计算
    print("\n[Test] ESDF Computation")
    
    # CPU版本
    from scipy.ndimage import distance_transform_edt
    start = time.time()
    esdf_cpu = distance_transform_edt(1 - grid)
    cpu_time = time.time() - start
    print(f"  CPU (scipy): {cpu_time*1000:.2f} ms")
    
    # GPU版本
    start = time.time()
    esdf_gpu = accelerator.compute_esdf(grid)
    gpu_time = time.time() - start
    print(f"  GPU: {gpu_time*1000:.2f} ms")
    print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
    
    # 测试射线投射
    print("\n[Test] Raycast")
    origin = np.array([500.0, 500.0]) * 0.1
    num_rays = 360
    angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    # GPU版本
    start = time.time()
    for _ in range(100):
        distances = accelerator.raycast(grid, origin, directions, 15.0, 0.1, 0.1)
    gpu_time = (time.time() - start) / 100
    print(f"  GPU (360 rays): {gpu_time*1000:.2f} ms")
    
    # 测试碰撞检测
    print("\n[Test] Collision Detection")
    positions = np.random.rand(64, 20, 2) * 100 * 0.1
    
    start = time.time()
    for _ in range(100):
        collision, dist = accelerator.check_collision(esdf_cpu, positions, 0.3, 0.1)
    gpu_time = (time.time() - start) / 100
    print(f"  GPU (64 trajectories × 20 points): {gpu_time*1000:.2f} ms")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
