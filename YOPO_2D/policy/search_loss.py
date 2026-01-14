"""
搜索代价损失函数 (Search Loss)

替代原有的目标引导损失 (GuidanceLoss)，使用不确定性作为搜索引导。

核心公式:
J_search = -Σ_{k=0}^K Δt * U(p(t_k))

引导无人机飞向高不确定性区域。
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg


class SearchLoss(nn.Module):
    """
    搜索损失 - 引导无人机飞向高不确定性区域
    
    公式: J_search = -Σ Δt * U(p_k)
    
    U(p) 越高 (不确定性越大)，代价越低（奖励探索）
    """
    
    def __init__(self):
        super().__init__()
        
        traj_cfg = cfg['trajectory']
        search_cfg = cfg.get('search', default={}) if isinstance(cfg._data, dict) else {}
        
        self.segment_time = traj_cfg['segment_time']
        self.num_eval_points = int(traj_cfg.get('collision_check_points', 20))
        
        # 搜索参数
        self.sensor_radius = float(search_cfg.get('sensor_radius', 5.0))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 不确定性地图 (训练时设置)
        self.uncertainty_maps = None
        self.map_info = None
    
    def set_uncertainty_maps(
        self,
        uncertainty_maps: list,
        map_info: dict = None
    ):
        """
        设置不确定性地图
        
        Args:
            uncertainty_maps: 不确定性地图列表 [(H, W), ...]
            map_info: {'size': [x, y], 'resolution': r}
        """
        if not uncertainty_maps:
            self.uncertainty_maps = None
            return
        
        self.uncertainty_maps = torch.stack([
            torch.tensor(m, dtype=torch.float32, device=self.device)
            for m in uncertainty_maps
        ], dim=0).unsqueeze(1)  # [M, 1, H, W]
        
        self.map_info = map_info or {
            'size': cfg['env']['map_size'],
            'resolution': cfg['env']['resolution']
        }
    
    def query_uncertainty(
        self,
        positions: torch.Tensor,
        map_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        查询位置的不确定性
        
        Args:
            positions: [batch, N, 2] 世界坐标
            map_idx: [batch] 地图索引
            
        Returns:
            uncertainties: [batch, N]
        """
        if self.uncertainty_maps is None:
            # 默认均匀不确定性
            return torch.ones(
                positions.shape[0], positions.shape[1],
                device=self.device
            )
        
        batch_size, N, _ = positions.shape
        resolution = self.map_info['resolution']
        
        # 获取地图尺寸
        maps = self.uncertainty_maps.squeeze(1)  # [M, H, W]
        map_count, h_x, w_y = maps.shape
        map_idx = map_idx % map_count
        
        # 网格坐标
        grid_x = (positions[:, :, 0] / resolution).clamp(0, h_x - 1.001)
        grid_y = (positions[:, :, 1] / resolution).clamp(0, w_y - 1.001)
        
        # 双线性插值
        x0 = torch.floor(grid_x).long()
        y0 = torch.floor(grid_y).long()
        x1 = (x0 + 1).clamp(max=h_x - 1)
        y1 = (y0 + 1).clamp(max=w_y - 1)
        
        wx = grid_x - x0.float()
        wy = grid_y - y0.float()
        
        # 展平索引
        maps_flat = maps.view(map_count, -1)  # [M, H*W]
        map_idx_exp = map_idx.view(-1, 1).expand(-1, N)
        
        idx00 = x0 * w_y + y0
        idx10 = x1 * w_y + y0
        idx01 = x0 * w_y + y1
        idx11 = x1 * w_y + y1
        
        v00 = maps_flat[map_idx_exp, idx00]
        v10 = maps_flat[map_idx_exp, idx10]
        v01 = maps_flat[map_idx_exp, idx01]
        v11 = maps_flat[map_idx_exp, idx11]
        
        uncertainties = (
            (1 - wx) * (1 - wy) * v00 +
            wx * (1 - wy) * v10 +
            (1 - wx) * wy * v01 +
            wx * wy * v11
        )
        
        return uncertainties
    
    def _poly5_positions(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor
    ) -> torch.Tensor:
        """
        使用5次多项式生成轨迹位置
        
        Args:
            start_state: [batch, 3, 2] = [[px, py], [vx, vy], [ax, ay]]
            end_state: [batch, 3, 2]
            
        Returns:
            positions: [batch, N, 2]
        """
        pos0 = start_state[:, 0, :]
        vel0 = start_state[:, 1, :]
        acc0 = start_state[:, 2, :]
        pos1 = end_state[:, 0, :]
        vel1 = end_state[:, 1, :]
        acc1 = end_state[:, 2, :]
        
        T = self.segment_time
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
        
        A = torch.tensor(
            [[T3, T4, T5],
             [3 * T2, 4 * T3, 5 * T4],
             [6 * T, 12 * T2, 20 * T3]],
            dtype=pos0.dtype,
            device=pos0.device
        )
        A_inv = torch.inverse(A)
        
        b = torch.stack([b0, b1, b2], dim=1)  # [B, 3, 2]
        a3_a5 = torch.matmul(A_inv, b)  # [B, 3, 2]
        a3, a4, a5 = a3_a5[:, 0, :], a3_a5[:, 1, :], a3_a5[:, 2, :]
        
        t = torch.linspace(0, T, self.num_eval_points, device=pos0.device)
        t1 = t.view(1, -1, 1)
        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t3 * t1
        t5 = t4 * t1
        
        pos = (
            a0.unsqueeze(1) +
            a1.unsqueeze(1) * t1 +
            a2.unsqueeze(1) * t2 +
            a3.unsqueeze(1) * t3 +
            a4.unsqueeze(1) * t4 +
            a5.unsqueeze(1) * t5
        )
        return pos
    
    def forward(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        map_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        计算搜索损失
        
        J_search = -Σ Δt * U(p_k)
        
        Args:
            start_state: [batch, 3, 2] 世界坐标系
            end_state: [batch, 3, 2] 世界坐标系
            map_idx: [batch] 地图索引
            
        Returns:
            cost: [batch] 搜索代价 (越低越好, 即高不确定性区域)
        """
        # 生成轨迹采样点
        positions = self._poly5_positions(start_state, end_state)  # [batch, N, 2]
        
        # 查询不确定性
        uncertainties = self.query_uncertainty(positions, map_idx)  # [batch, N]
        
        # 时间步长
        dt = self.segment_time / self.num_eval_points
        
        # 搜索代价 = -Σ Δt * U (负号: 高不确定性 = 低代价)
        # 为了与其他损失一致（都是越小越好），我们返回负的不确定性积分
        search_cost = -dt * uncertainties.sum(dim=-1)
        
        return search_cost


class InformationGainLoss(nn.Module):
    """
    信息增益损失 - 更高级的搜索损失
    
    考虑传感器覆盖范围内的总信息增益
    """
    
    def __init__(self):
        super().__init__()
        
        traj_cfg = cfg['trajectory']
        search_cfg = cfg.get('search', default={}) if isinstance(cfg._data, dict) else {}
        
        self.segment_time = traj_cfg['segment_time']
        self.num_eval_points = int(traj_cfg.get('collision_check_points', 10))
        self.sensor_radius = float(search_cfg.get('sensor_radius', 5.0))
        self.resolution = cfg['env']['resolution']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.uncertainty_maps = None
        self.map_info = None
    
    def set_uncertainty_maps(self, uncertainty_maps: list, map_info: dict = None):
        """设置不确定性地图"""
        if not uncertainty_maps:
            self.uncertainty_maps = None
            return
        
        self.uncertainty_maps = torch.stack([
            torch.tensor(m, dtype=torch.float32, device=self.device)
            for m in uncertainty_maps
        ], dim=0)  # [M, H, W]
        
        self.map_info = map_info or {
            'size': cfg['env']['map_size'],
            'resolution': cfg['env']['resolution']
        }
    
    def forward(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        map_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        计算信息增益损失
        
        考虑传感器覆盖范围内的期望信息增益
        """
        if self.uncertainty_maps is None:
            return torch.zeros(start_state.shape[0], device=self.device)
        
        batch_size = start_state.shape[0]
        
        # 简化版: 只考虑轨迹终点附近的不确定性
        end_pos = end_state[:, 0, :]  # [batch, 2]
        
        # 采样传感器覆盖范围内的点
        num_samples = 16
        angles = torch.linspace(0, 2 * np.pi, num_samples, device=self.device)
        radii = torch.linspace(0, self.sensor_radius, 4, device=self.device)
        
        total_uncertainty = torch.zeros(batch_size, device=self.device)
        
        for r in radii:
            for a in angles:
                offset = torch.tensor([r * torch.cos(a), r * torch.sin(a)], device=self.device)
                sample_pos = end_pos + offset
                
                # 查询不确定性 (简化版)
                grid_x = (sample_pos[:, 0] / self.resolution).clamp(0, self.uncertainty_maps.shape[1] - 1).long()
                grid_y = (sample_pos[:, 1] / self.resolution).clamp(0, self.uncertainty_maps.shape[2] - 1).long()
                
                map_idx_clamped = map_idx % self.uncertainty_maps.shape[0]
                uncertainty = self.uncertainty_maps[map_idx_clamped, grid_x, grid_y]
                total_uncertainty += uncertainty
        
        # 归一化
        total_uncertainty /= (num_samples * len(radii))
        
        # 返回负值 (高不确定性 = 低代价)
        return -total_uncertainty


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 测试搜索损失
    print("Testing search loss...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试不确定性地图
    H, W = 100, 100
    uncertainty_map = np.ones((H, W), dtype=np.float32)
    # 在某些区域设置低不确定性 (已探索)
    uncertainty_map[20:40, 20:40] = 0.1
    uncertainty_map[60:80, 60:80] = 0.1
    
    # 创建损失函数
    loss_fn = SearchLoss()
    loss_fn.set_uncertainty_maps(
        [uncertainty_map],
        {'size': [100, 100], 'resolution': 1.0}
    )
    
    # 测试数据
    batch_size = 4
    start_state = torch.zeros(batch_size, 3, 2, device=device)
    start_state[:, 0, :] = torch.tensor([[30, 30], [70, 70], [50, 50], [10, 10]], device=device).float()
    start_state[:, 1, :] = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]], device=device).float()
    
    end_state = torch.zeros(batch_size, 3, 2, device=device)
    end_state[:, 0, :] = torch.tensor([[35, 30], [75, 70], [55, 50], [15, 10]], device=device).float()
    end_state[:, 1, :] = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]], device=device).float()
    
    map_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    cost = loss_fn(start_state, end_state, map_idx)
    print(f"Search costs: {cost}")
    print("(Negative = reward, lower value = higher uncertainty = better)")
    
    # 可视化
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(uncertainty_map.T, origin='lower', cmap='hot', vmin=0, vmax=1)
    
    for i in range(batch_size):
        s = start_state[i, 0].cpu().numpy()
        e = end_state[i, 0].cpu().numpy()
        ax.plot([s[0], e[0]], [s[1], e[1]], 'b-o', linewidth=2)
        ax.annotate(f'Cost={cost[i].item():.2f}', (e[0], e[1]), fontsize=10, color='white')
    
    ax.set_title('Uncertainty Map and Trajectory Costs')
    plt.colorbar(im, ax=ax, label='Uncertainty')
    plt.savefig('search_loss_test.png')
    print("Saved to search_loss_test.png")
    plt.show()
