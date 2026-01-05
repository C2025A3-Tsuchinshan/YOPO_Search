"""
2D YOPO 损失函数 (对齐 YOPO_Sim/YOPO/loss/)

关键对齐:
- 使用5次多项式的QP矩阵计算Jerk/Acc积分 (对齐 YOPO_Sim smoothness_loss.py)
- 使用 exp(-(d-d0)/r) 安全代价 (对齐 YOPO_Sim safety_loss.py)
- 使用投影相似度引导损失 (对齐 YOPO_Sim guidance_loss.py)
- 权重归一化 (对齐 YOPO_Sim loss_function.py denormalize_weight)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg


class QPMatrices2D:
    """
    2D版本的QP矩阵生成 (对齐 YOPO_Sim YOPOLoss.qp_generation)
    
    用于5次多项式轨迹的Jerk和Acc积分计算
    """
    
    def __init__(self, segment_time: float):
        self.sgm_time = segment_time
        self._C, self._B, self._L, self._RJ, self._RA = self._qp_generation()
    
    def _qp_generation(self):
        """生成QP矩阵 (对齐 YOPO_Sim)"""
        T = self.sgm_time
        
        # 映射矩阵 A: 从多项式系数到边界条件
        A = torch.zeros((6, 6))
        for i in range(3):
            A[2 * i, i] = math.factorial(i)
            for j in range(i, 6):
                A[2 * i + 1, j] = math.factorial(j) / math.factorial(j - i) * (T ** (j - i))
        
        # H: Jerk海森矩阵
        H = torch.zeros((6, 6))
        for i in range(3, 6):
            for j in range(3, 6):
                H[i, j] = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * (T ** (i + j - 5))
        
        # Q: Acc海森矩阵
        Q = torch.zeros((6, 6))
        for i in range(2, 6):
            for j in range(2, 6):
                Q[i, j] = (i * (i - 1)) * (j * (j - 1)) / (i + j - 3) * (T ** (i + j - 3))
        
        return self._stack_opt_dep(A, H, Q)
    
    def _stack_opt_dep(self, A, H, Q):
        """堆叠优化依赖矩阵"""
        Ct = torch.zeros((6, 6))
        Ct[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1
        
        _C = torch.transpose(Ct, 0, 1)
        B = torch.inverse(A)
        B_T = torch.transpose(B, 0, 1)
        _L = B @ Ct
        _R_Jerk = _C @ B_T @ H @ B @ Ct
        _R_Acc = _C @ B_T @ Q @ B @ Ct
        
        return _C, B, _L, _R_Jerk, _R_Acc
    
    def to(self, device):
        """移动到指定设备"""
        self._L = self._L.to(device)
        self._RJ = self._RJ.to(device)
        self._RA = self._RA.to(device)
        return self


class SafetyLoss2D(nn.Module):
    """安全损失 - 基于ESDF的避障"""
    
    def __init__(self, esdf_maps: list = None):
        """
        Args:
            esdf_maps: 预计算的ESDF地图列表 [(H, W), ...]
        """
        super().__init__()
        
        traj_cfg = cfg['trajectory']
        
        self.d0 = traj_cfg['safe_distance']
        self.robot_r = traj_cfg['robot_inflation']
        self.segment_time = traj_cfg['segment_time']
        self.num_eval_points = 20
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ESDF相关
        self.esdf_maps = None
        self.map_info = None  # {'size': [x, y], 'resolution': r}
        
        if esdf_maps is not None:
            self.set_esdf_maps(esdf_maps)
    
    def set_esdf_maps(self, esdf_maps: list, map_info: dict = None):
        """设置ESDF地图"""
        if not esdf_maps:
            self.esdf_maps = None
        else:
            esdf_tensor = torch.stack(
                [torch.tensor(m, dtype=torch.float32, device=self.device) for m in esdf_maps],
                dim=0
            )
            self.esdf_maps = esdf_tensor.unsqueeze(1)  # [M, 1, H, W]
        self.map_info = map_info or {
            'size': cfg['env']['map_size'],
            'resolution': cfg['env']['resolution']
        }
    
    def query_distance(self, positions: torch.Tensor, map_idx: torch.Tensor) -> torch.Tensor:
        """
        查询位置到最近障碍物的距离
        
        Args:
            positions: [batch, N, 2] 世界坐标
            map_idx: [batch] 地图索引
            
        Returns:
            distances: [batch, N] 距离值 (meters)
        """
        batch_size, N, _ = positions.shape
        
        if self.esdf_maps is None:
            return torch.ones(batch_size, N, device=self.device) * self.d0 * 2
        
        resolution = self.map_info['resolution']

        maps = self.esdf_maps.squeeze(1)  # [M, H, W], H: x, W: y
        map_count, h_x, w_y = maps.shape
        map_idx = map_idx % map_count

        grid_x = (positions[:, :, 0] / resolution).clamp(0, h_x - 1)
        grid_y = (positions[:, :, 1] / resolution).clamp(0, w_y - 1)

        x0 = torch.floor(grid_x).long()
        y0 = torch.floor(grid_y).long()
        x1 = (x0 + 1).clamp(max=h_x - 1)
        y1 = (y0 + 1).clamp(max=w_y - 1)

        wx = (grid_x - x0.float())
        wy = (grid_y - y0.float())

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

        distances = (
            (1 - wx) * (1 - wy) * v00 +
            wx * (1 - wy) * v10 +
            (1 - wx) * wy * v01 +
            wx * wy * v11
        )

        return distances * resolution
    
    def compute_cost(self, distance: torch.Tensor) -> torch.Tensor:
        """
        计算安全代价
        
        使用三段式代价函数:
        - d > d0: 0
        - r < d <= d0: (d - d0)^2
        - d <= r: (d - d0)^2 + (r - d)^3 / 3
        """
        d0 = self.d0
        r = self.robot_r
        
        cost = torch.zeros_like(distance)
        
        # 中间区域
        mask_mid = (distance > r) & (distance <= d0)
        cost[mask_mid] = (distance[mask_mid] - d0) ** 2
        
        # 危险区域
        mask_danger = distance <= r
        cost[mask_danger] = (distance[mask_danger] - d0) ** 2 + (r - distance[mask_danger]) ** 3 / 3
        
        return cost
    
    def compute_cost(self, distance: torch.Tensor) -> torch.Tensor:
        """
        YOPO_Sim风格的安全代价: exp(-(d - d0)/r)
        """
        return torch.exp(-(distance - self.d0) / self.robot_r)

    def _poly5_positions(self, start_state: torch.Tensor, end_state: torch.Tensor) -> torch.Tensor:
        """
        使用5次多项式生成轨迹位置，保证与test.py的Poly5Solver2D一致
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
        Safety loss with Poly5 sampling and YOPO_Sim-style cost.
        """
        positions = self._poly5_positions(start_state, end_state)  # [batch, N, 2]
        distances = self.query_distance(positions, map_idx)  # [batch, N]
        costs = self.compute_cost(distances)  # [batch, N]
        return costs.mean(dim=-1)

    def _forward_hermite(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        map_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        计算安全损失
        
        Args:
            start_state: [batch, 3, 2] = [[px, py], [vx, vy], [ax, ay]]
            end_state: [batch, 3, 2]
            map_idx: [batch] 地图索引
            
        Returns:
            cost: [batch] 安全代价
        """
        batch_size = start_state.shape[0]
        
        # 生成轨迹采样点
        t = torch.linspace(0, self.segment_time, self.num_eval_points, device=self.device)
        t = t.view(1, -1, 1)  # [1, N, 1]
        
        # 五次多项式系数 (简化: 使用线性插值)
        alpha = t / self.segment_time  # [1, N, 1]
        
        # 插值位置
        start_pos = start_state[:, 0:1, :]  # [batch, 1, 2]
        end_pos = end_state[:, 0:1, :]  # [batch, 1, 2]
        
        # 使用三次Hermite插值
        start_vel = start_state[:, 1:2, :]
        end_vel = end_state[:, 1:2, :]
        
        # H00, H10, H01, H11
        h00 = 2*alpha**3 - 3*alpha**2 + 1
        h10 = alpha**3 - 2*alpha**2 + alpha
        h01 = -2*alpha**3 + 3*alpha**2
        h11 = alpha**3 - alpha**2
        
        positions = (h00 * start_pos + h10 * self.segment_time * start_vel + 
                    h01 * end_pos + h11 * self.segment_time * end_vel)  # [batch, N, 2]
        
        # 查询距离
        distances = self.query_distance(positions, map_idx)  # [batch, N]
        
        # 计算代价
        costs = self.compute_cost(distances)  # [batch, N]
        
        # 平均代价
        return costs.mean(dim=-1)  # [batch]


class SmoothnessLoss2D(nn.Module):
    """
    平滑性损失 (对齐 YOPO_Sim SmoothnessLoss)
    
    使用QP矩阵精确计算Jerk和Acc积分
    """
    
    def __init__(self):
        super().__init__()
        traj_cfg = cfg['trajectory']
        self.segment_time = traj_cfg['segment_time']
        
        # 初始化QP矩阵
        self.qp = QPMatrices2D(self.segment_time)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qp.to(self.device)
    
    def forward(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor
    ) -> tuple:
        """
        计算平滑性损失 (对齐 YOPO_Sim SmoothnessLoss.forward)
        
        Args:
            start_state: [batch, 3, 2] = [[px, py], [vx, vy], [ax, ay]]
            end_state: [batch, 3, 2]
            
        Returns:
            jerk_cost, acc_cost: [batch]
        """
        # 转换为 [batch, 2, 3] -> [px, vx, ax; py, vy, ay] (对齐 YOPO_Sim Df/Dp 格式)
        Df = start_state.permute(0, 2, 1)  # [batch, 2, 3]
        Dp = end_state.permute(0, 2, 1)  # [batch, 2, 3]
        
        batch_size = Df.shape[0]
        RJ = self.qp._RJ.to(Df.device).unsqueeze(0).expand(batch_size, -1, -1)
        RA = self.qp._RA.to(Df.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # D_all: [batch, 2, 6] = [px_start, vx_start, ax_start, px_end, vx_end, ax_end; ...]
        D_all = torch.cat([Df, Dp], dim=2)
        
        # 分离 x, y
        dx = D_all[:, 0].unsqueeze(2)  # [batch, 6, 1]
        dy = D_all[:, 1].unsqueeze(2)  # [batch, 6, 1]
        
        # Jerk 代价: dx^T @ RJ @ dx + dy^T @ RJ @ dy
        jerk_cost = (
            dx.transpose(1, 2) @ RJ @ dx +
            dy.transpose(1, 2) @ RJ @ dy
        ).squeeze()  # [batch]
        
        # Acc 代价
        acc_cost = (
            dx.transpose(1, 2) @ RA @ dx +
            dy.transpose(1, 2) @ RA @ dy
        ).squeeze()  # [batch]
        
        return jerk_cost, acc_cost


class Poly5SmoothnessLoss2D(nn.Module):
    """Poly5采样的Jerk/曲率损失"""

    def __init__(self):
        super().__init__()
        traj_cfg = cfg['trajectory']
        self.segment_time = traj_cfg['segment_time']
        self.num_eval_points = int(traj_cfg.get('smoothness_eval_points', 20))
        if self.num_eval_points < 5:
            self.num_eval_points = 5
        self.eps = 1e-6

    def _poly5_coeffs(self, start_state: torch.Tensor, end_state: torch.Tensor) -> tuple:
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
        return a0, a1, a2, a3, a4, a5

    def forward(self, start_state: torch.Tensor, end_state: torch.Tensor) -> tuple:
        a0, a1, a2, a3, a4, a5 = self._poly5_coeffs(start_state, end_state)

        t = torch.linspace(0, self.segment_time, self.num_eval_points, device=a0.device, dtype=a0.dtype)
        t1 = t.view(1, -1, 1)
        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t3 * t1

        v = (
            a1.unsqueeze(1) +
            2 * a2.unsqueeze(1) * t1 +
            3 * a3.unsqueeze(1) * t2 +
            4 * a4.unsqueeze(1) * t3 +
            5 * a5.unsqueeze(1) * t4
        )

        acc = (
            2 * a2.unsqueeze(1) +
            6 * a3.unsqueeze(1) * t1 +
            12 * a4.unsqueeze(1) * t2 +
            20 * a5.unsqueeze(1) * t3
        )

        jerk = (
            6 * a3.unsqueeze(1) +
            24 * a4.unsqueeze(1) * t1 +
            60 * a5.unsqueeze(1) * t2
        )

        jerk_sq = (jerk ** 2).sum(dim=-1)
        jerk_cost = jerk_sq.mean(dim=-1) * self.segment_time

        speed = torch.norm(v, dim=-1).clamp(min=self.eps)
        cross = v[..., 0] * acc[..., 1] - v[..., 1] * acc[..., 0]
        curvature = torch.abs(cross) / (speed ** 3 + self.eps)
        curvature_cost = curvature.mean(dim=-1)

        return jerk_cost, curvature_cost


class AccelerationLoss2D(nn.Module):
    """加速度损失"""
    
    def __init__(self):
        super().__init__()
        self.segment_time = cfg['trajectory']['segment_time']
        self.max_acc = cfg['robot']['max_acc']
    
    def forward(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor
    ) -> torch.Tensor:
        """
        计算加速度损失
        
        Args:
            start_state: [batch, 3, 2]
            end_state: [batch, 3, 2]
            
        Returns:
            cost: [batch]
        """
        start_acc = start_state[:, 2, :]
        end_acc = end_state[:, 2, :]
        
        # 加速度平方的平均
        acc_sq = (start_acc ** 2 + end_acc ** 2) / 2
        cost = acc_sq.sum(dim=-1)
        
        return cost


class GuidanceLoss2D(nn.Module):
    """
    引导损失 (对齐 YOPO_Sim GuidanceLoss)
    
    使用投影相似度: 轨迹在目标方向上的投影长度
    横向容差: 允许一定程度的侧向探索，避免陷入局部最优
    轨迹长度奖励: 鼓励更长的轨迹，避免遇到障碍物时过度缩短
    """
    
    def __init__(self):
        super().__init__()
        train_cfg = cfg['training']
        traj_cfg = cfg['trajectory']
        self.radio_range = float(traj_cfg.get('radio_range', traj_cfg['planning_horizon'] / 2))
        self.goal_length = float(train_cfg.get('goal_length', 2.0 * self.radio_range))
        self.vel_dir_weight = 0  # 可选: 末端速度方向约束
        # 横向容差权重: 越小越允许侧向探索 (对齐 YOPO_Sim perp_weight)
        self.perp_weight = float(traj_cfg.get('guidance_perp_weight', 0.5))
        # 轨迹长度奖励权重: 惩罚过短轨迹，鼓励绕行
        self.length_reward_weight = float(traj_cfg.get('guidance_length_weight', 0.1))
    
    def forward(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        goal: torch.Tensor
    ) -> torch.Tensor:
        """
        计算引导损失 (对齐 YOPO_Sim GuidanceLoss.forward)
        
        Args:
            start_state: [batch, 3, 2]
            end_state: [batch, 3, 2]
            goal: [batch, 2] 世界坐标系下的目标
            
        Returns:
            cost: [batch]
        """
        cur_pos = start_state[:, 0, :]  # [batch, 2]
        end_pos = end_state[:, 0, :]  # [batch, 2]
        end_vel = end_state[:, 1, :]  # [batch, 2]
        
        traj_dir = end_pos - cur_pos  # [batch, 2]
        goal_dir = goal - cur_pos  # [batch, 2]
        
        # 使用相似度损失 (对齐 YOPO_Sim similarity_loss)
        guidance_loss = self._similarity_loss(traj_dir, goal_dir)
        
        # 轨迹长度惩罚: 惩罚过短轨迹，鼓励绕行而不是缩短
        if self.length_reward_weight > 0:
            traj_length = traj_dir.norm(dim=1)  # [B]
            max_length = self.radio_range * 2
            # 短轨迹惩罚: (1 - length/max_length) 越短惩罚越大
            length_penalty = (1.0 - traj_length / max_length).clamp(min=0)
            guidance_loss = guidance_loss + self.length_reward_weight * length_penalty * max_length
        
        if self.vel_dir_weight > 0:
            vel_dir_loss = self._derivative_similarity_loss(end_vel, goal_dir)
            guidance_loss = guidance_loss + self.vel_dir_weight * vel_dir_loss
        
        return guidance_loss
    
    def _similarity_loss(self, traj_dir: torch.Tensor, goal_dir: torch.Tensor) -> torch.Tensor:
        """
        投影相似度损失 (对齐 YOPO_Sim GuidanceLoss.similarity_loss)
        
        更高的余弦相似度和更长的轨迹被优先
        横向容差: perp_weight 越小，越允许侧向探索
        """
        goal_dir_norm = goal_dir / (goal_dir.norm(dim=1, keepdim=True) + 1e-8)  # [B, 2]
        
        # 轨迹在目标方向上的投影长度
        traj_along = (traj_dir * goal_dir_norm).sum(dim=1)  # [B]
        goal_length = goal_dir.norm(dim=1)  # [B]
        
        # 沿目标方向的长度差 (余弦相似度)
        parallel_diff = (goal_length - traj_along).abs()  # [B]
        
        # 垂直于目标方向的分量
        traj_perp = traj_dir - traj_along.unsqueeze(1) * goal_dir_norm  # [B, 2]
        perp_diff = traj_perp.norm(dim=1)  # [B]
        
        # 使用配置的横向容差权重 (对齐 YOPO_Sim: perp_weight 默认 0.5)
        similarity_loss = parallel_diff + self.perp_weight * perp_diff
        return similarity_loss
    
    def _derivative_similarity_loss(self, derivative: torch.Tensor, goal_dir: torch.Tensor) -> torch.Tensor:
        """约束速度方向朝向目标"""
        goal_dir_norm = goal_dir / (goal_dir.norm(dim=1, keepdim=True) + 1e-8)
        derivative_norm = derivative / (derivative.norm(dim=1, keepdim=True) + 1e-8)
        similarity = (derivative_norm * goal_dir_norm).sum(dim=1)
        return 1 - similarity
        goal_dist = torch.norm(goal_dir, dim=-1, keepdim=True).clamp(min=1e-6)
        goal_dir_norm = goal_dir / goal_dist
        
        # 投影损失: 轨迹在目标方向上的投影
        projection = (traj_dir * goal_dir_norm).sum(dim=-1)  # [batch]
        
        # 我们希望投影越大越好, 所以损失为负投影
        # 同时考虑垂直分量
        perp = traj_dir - projection.unsqueeze(-1) * goal_dir_norm
        perp_dist = torch.norm(perp, dim=-1)
        
        # 组合损失
        parallel_loss = (goal_dist.squeeze(-1).clamp(max=self.planning_horizon * 2) - projection).abs()
        perp_loss = perp_dist
        
        cost = parallel_loss + 0.5 * perp_loss
        
        return cost


class DirectionLoss2D(nn.Module):
    """方向一致性损失 - 约束末端速度/加速度沿轨迹方向"""

    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor
    ) -> torch.Tensor:
        start_pos = start_state[:, 0, :]
        end_pos = end_state[:, 0, :]
        traj_dir = end_pos - start_pos
        traj_norm = torch.norm(traj_dir, dim=-1, keepdim=True).clamp(min=self.eps)
        traj_unit = traj_dir / traj_norm

        end_vel = end_state[:, 1, :]
        end_acc = end_state[:, 2, :]
        vel_norm = torch.norm(end_vel, dim=-1, keepdim=True).clamp(min=self.eps)
        acc_norm = torch.norm(end_acc, dim=-1, keepdim=True).clamp(min=self.eps)
        vel_unit = end_vel / vel_norm
        acc_unit = end_acc / acc_norm

        vel_cos = (vel_unit * traj_unit).sum(dim=-1)
        acc_cos = (acc_unit * traj_unit).sum(dim=-1)

        vel_mask = (vel_norm.squeeze(-1) > 1e-3).float()
        acc_mask = (acc_norm.squeeze(-1) > 1e-3).float()

        vel_loss = (1.0 - vel_cos) * vel_mask
        acc_loss = (1.0 - acc_cos) * acc_mask

        return vel_loss + 0.5 * acc_loss


class YopoLoss2D(nn.Module):
    """
    YOPO 2D 综合损失 (对齐 YOPO_Sim YOPOLoss)
    
    权重归一化: 按速度缩放以保证不同速度下的一致性
    """
    
    def __init__(self, esdf_maps: list = None):
        super().__init__()
        
        traj_cfg = cfg['trajectory']
        train_cfg = cfg['training']
        robot_cfg = cfg['robot']
        
        # 原始权重 (从配置读取)
        self.w_guidance = float(traj_cfg['w_guidance'])
        self.w_smoothness = float(traj_cfg['w_smoothness'])
        self.w_safety = float(traj_cfg['w_safety'])
        self.w_acceleration = float(traj_cfg['w_acceleration'])
        self.w_direction = float(traj_cfg.get('w_direction', 0.0))
        
        # 子损失模块
        self.safety_loss = SafetyLoss2D(esdf_maps)
        self.smoothness_loss = SmoothnessLoss2D()
        self.guidance_loss = GuidanceLoss2D()
        self.direction_loss = DirectionLoss2D()
        
        # 权重归一化 (对齐 YOPO_Sim denormalize_weight)
        self._normalize_weights()
        
        print("---------- 2D Loss Weights (normalized) ----------")
        print(f"| {'guidance':<14} = {self.w_guidance:>8.4f} |")
        print(f"| {'smoothness':<14} = {self.w_smoothness:>8.4f} |")
        print(f"| {'acceleration':<14} = {self.w_acceleration:>8.4f} |")
        print(f"| {'safety':<14} = {self.w_safety:>8.4f} |")
        print(f"| {'direction':<14} = {self.w_direction:>8.4f} |")
        print("-" * 50)
    
    def _normalize_weights(self):
        """
        根据速度归一化权重 (对齐 YOPO_Sim denormalize_weight)
        
        - smoothness cost: 时间积分 jerk² → 速度缩放 n 倍时，代价缩放 n^5 倍
        - acceleration cost: 时间积分 acc² → 速度缩放 n 倍时，代价缩放 n^3 倍
        - safety cost: 时间积分 → 速度缩放 n 倍时，代价缩放 1/n 倍 (但这里用恒定权重)
        - guidance cost: 与速度无关
        """
        train_cfg = cfg['training']
        robot_cfg = cfg['robot']
        vel_max_train = float(train_cfg.get('vel_max_train', robot_cfg['max_vel']))
        vel_scale = vel_max_train / 1.0
        
        # 对齐 YOPO_Sim
        self.w_smoothness = self.w_smoothness / (vel_scale ** 5)
        self.w_acceleration = self.w_acceleration / (vel_scale ** 3)
        # safety 和 guidance 权重保持不变 (对齐 YOPO_Sim)
    
    def set_esdf_maps(self, esdf_maps: list, map_info: dict = None):
        """设置ESDF地图"""
        self.safety_loss.set_esdf_maps(esdf_maps, map_info)
    
    def forward(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        goal: torch.Tensor,
        map_idx: torch.Tensor
    ) -> tuple:
        """
        计算总损失 (对齐 YOPO_Sim YOPOLoss.forward)
        
        Args:
            start_state: [batch, 3, 2] 世界坐标系 [[px, py], [vx, vy], [ax, ay]]
            end_state: [batch, 3, 2] 世界坐标系
            goal: [batch, 2] 世界坐标系
            map_idx: [batch]
            
        Returns:
            total_cost: [batch]
            costs_dict: 各项损失的字典
        """
        # 安全损失 (对齐 YOPO_Sim SafetyLoss)
        safety_cost = self.safety_loss(start_state, end_state, map_idx)
        
        # 平滑损失 (对齐 YOPO_Sim SmoothnessLoss: QP矩阵计算)
        jerk_cost, acc_cost = self.smoothness_loss(start_state, end_state)
        
        # 引导损失 (对齐 YOPO_Sim GuidanceLoss)
        guidance_cost = self.guidance_loss(start_state, end_state, goal)
        
        # 方向一致性损失 (可选)
        direction_cost = self.direction_loss(start_state, end_state) if self.w_direction > 0 else torch.zeros_like(safety_cost)
        
        # 总损失 (对齐 YOPO_Sim: w_s * smooth + w_c * safety + w_g * goal + w_a * acc)
        total_cost = (
            self.w_smoothness * jerk_cost +
            self.w_acceleration * acc_cost +
            self.w_safety * safety_cost +
            self.w_guidance * guidance_cost +
            self.w_direction * direction_cost
        )
        
        return total_cost, {
            'safety': safety_cost,
            'smoothness': jerk_cost,
            'acceleration': acc_cost,
            'guidance': guidance_cost,
            'direction': direction_cost
        }


if __name__ == "__main__":
    # 测试损失函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loss_fn = YopoLoss2D()
    
    batch_size = 4
    start_state = torch.randn(batch_size, 3, 2, device=device)
    end_state = torch.randn(batch_size, 3, 2, device=device)
    goal = torch.randn(batch_size, 2, device=device) * 10
    map_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    total_cost, costs_dict = loss_fn(start_state, end_state, goal, map_idx)
    
    print(f"Total cost: {total_cost}")
    print(f"Costs dict: {costs_dict}")
