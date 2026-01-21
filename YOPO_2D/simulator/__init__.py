"""
Simulator module
"""
from simulator.map_generator import Map2D, MapManager
from simulator.sensor import Lidar2D, SensorSimulator

# CUDA加速模块 (可选)
try:
    from simulator.cuda_accelerator import (
        ESDFCUDA,
        RaycastCUDA,
        CollisionCheckerCUDA,
        TrajectoryGeneratorCUDA,
        YOPO2DCUDAAccelerator,
        get_accelerator,
        CUDA_AVAILABLE,
        CUPY_AVAILABLE
    )
except ImportError:
    CUDA_AVAILABLE = False
    CUPY_AVAILABLE = False

__all__ = [
    'Map2D', 'MapManager', 'Lidar2D', 'SensorSimulator',
    # CUDA
    'ESDFCUDA', 'RaycastCUDA', 'CollisionCheckerCUDA', 
    'TrajectoryGeneratorCUDA', 'YOPO2DCUDAAccelerator',
    'get_accelerator', 'CUDA_AVAILABLE', 'CUPY_AVAILABLE'
]
