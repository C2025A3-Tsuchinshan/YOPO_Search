#!/usr/bin/env python
"""
CUDA加速模块性能测试
"""
import torch
import numpy as np
import time

# 测试CUDA加速模块
from simulator.cuda_accelerator import (
    CUDA_AVAILABLE, CUPY_AVAILABLE,
    ESDFCUDA, RaycastCUDA, CollisionCheckerCUDA
)

def main():
    print('='*60)
    print('YOPO_2D CUDA Accelerator Performance Test')
    print('='*60)

    # 1. 环境信息
    print(f'\n[环境信息]')
    print(f'  CUDA Available: {CUDA_AVAILABLE}')
    if CUDA_AVAILABLE:
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CuPy Available: {CUPY_AVAILABLE}')

    # 2. 测试ESDF
    print(f'\n[ESDF计算测试]')

    # 测试不同尺寸的地图
    for grid_size in [500, 1000, 2000]:
        # 创建测试地图
        np.random.seed(42)
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        # 添加随机障碍
        n_obstacles = int(grid_size * 0.1)
        for _ in range(n_obstacles):
            x, y = np.random.randint(50, grid_size-50, 2)
            r = np.random.randint(10, 30)
            xx, yy = np.ogrid[:grid_size, :grid_size]
            mask = (xx-x)**2 + (yy-y)**2 < r**2
            grid[mask] = 1

        esdf_cuda = ESDFCUDA()

        # CPU计时
        from scipy.ndimage import distance_transform_edt
        start = time.time()
        esdf_cpu = distance_transform_edt(1 - grid)
        cpu_time = (time.time() - start) * 1000

        # GPU计时
        if CUDA_AVAILABLE:
            # 预热
            _ = esdf_cuda.compute_esdf(grid)
            torch.cuda.synchronize()
            
            start = time.time()
            esdf_gpu = esdf_cuda.compute_esdf(grid)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) * 1000
            
            print(f'  Grid Size: {grid_size}x{grid_size}')
            print(f'    CPU (scipy): {cpu_time:.2f}ms')
            print(f'    GPU (torch): {gpu_time:.2f}ms')
            print(f'    Speedup: {cpu_time/gpu_time:.2f}x')
        else:
            print(f'  Grid Size: {grid_size}x{grid_size}')
            print(f'    CPU (scipy): {cpu_time:.2f}ms')

    # 使用最后生成的地图继续后续测试
    grid = np.zeros((500, 500), dtype=np.float32)
    np.random.seed(42)
    for _ in range(50):
        x, y = np.random.randint(50, 450, 2)
        r = np.random.randint(10, 30)
        xx, yy = np.ogrid[:500, :500]
        mask = (xx-x)**2 + (yy-y)**2 < r**2
        grid[mask] = 1
    esdf_gpu = esdf_cuda.compute_esdf(grid) if CUDA_AVAILABLE else esdf_cpu

    # 3. 测试Raycast
    print(f'\n[射线投射测试]')
    raycast = RaycastCUDA()
    grid_t = torch.from_numpy(grid).float().to(raycast.device)
    origin = torch.tensor([250*0.1, 250*0.1], device=raycast.device)

    # 生成360度射线
    n_rays = 360
    angles = torch.linspace(0, 2*np.pi, n_rays, device=raycast.device)
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    if CUDA_AVAILABLE:
        # 预热
        _ = raycast.raycast_batch_torch(grid_t, origin, directions, max_range=10.0)
        torch.cuda.synchronize()

        start = time.time()
        distances = raycast.raycast_batch_torch(grid_t, origin, directions, max_range=10.0)
        torch.cuda.synchronize()
        ray_time = (time.time() - start) * 1000

        print(f'  Rays: {n_rays}')
        print(f'  GPU Time: {ray_time:.2f}ms')
        print(f'  Per Ray: {ray_time/n_rays*1000:.2f}us')
    else:
        print(f'  CUDA not available for raycast test')

    # 4. 测试碰撞检测
    print(f'\n[碰撞检测测试]')
    collision = CollisionCheckerCUDA()

    if CUDA_AVAILABLE:
        # 准备ESDF
        esdf_t = torch.from_numpy(esdf_gpu if isinstance(esdf_gpu, np.ndarray) else esdf_cpu).float().to(collision.device)

        # 生成测试轨迹
        n_trajs = 64
        n_points = 20
        trajectories = torch.rand(n_trajs, n_points, 2, device=collision.device) * 40 + 5  # 世界坐标
        robot_radius = 0.5
        resolution = 0.1

        # 预热
        _ = collision.check_trajectory_collision(esdf_t, trajectories, robot_radius, resolution)
        torch.cuda.synchronize()

        start = time.time()
        collision_mask = collision.check_trajectory_collision(esdf_t, trajectories, robot_radius, resolution)
        torch.cuda.synchronize()
        col_time = (time.time() - start) * 1000

        # 计算安全轨迹数量（没有任何碰撞点的轨迹）
        safe_mask = ~collision_mask.any(dim=1)

        print(f'  Trajectories: {n_trajs}, Points/Traj: {n_points}')
        print(f'  GPU Time: {col_time:.2f}ms')
        print(f'  Per Trajectory: {col_time/n_trajs*1000:.2f}us')
        print(f'  Safe Trajectories: {safe_mask.sum().item()}/{n_trajs}')
    else:
        print(f'  CUDA not available for collision test')

    print('\n' + '='*60)
    print('Test Complete!')
    print('='*60)

if __name__ == '__main__':
    main()
