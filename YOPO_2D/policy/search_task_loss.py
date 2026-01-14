"""
搜索任务专用损失函数 (YopoSearchLoss)

替代原有的 YopoLoss2D，使用不确定性搜索代价替代目标引导代价。

损失组成:
- J_search: 搜索引导损失 (引导无人机飞向高不确定性区域)
- J_coll: 避障损失 (基于ESDF)
- J_smooth: 平滑性损失 (Jerk积分)
- J_acc: 加速度损失

总损失: L = w_search * J_search + w_safety * J_coll + w_smooth * J_smooth + w_acc * J_acc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from policy.loss import SafetyLoss2D, SmoothnessLoss2D, QPMatrices2D
from policy.search_loss import SearchLoss


class YopoSearchLoss(nn.Module):
    """
    搜索任务综合损失函数
    
    与 YopoLoss2D 的区别:
    - 用 SearchLoss (不确定性) 替代 GuidanceLoss (目标引导)
    - 不需要显式的目标点输入
    """
    
    def __init__(
        self,
        esdf_maps: list = None,
        uncertainty_maps: list = None
    ):
        super().__init__()
        
        traj_cfg = cfg['trajectory']
        train_cfg = cfg['training']
        robot_cfg = cfg['robot']
        search_cfg = cfg.get('search', default={}) if isinstance(cfg._data, dict) else {}
        
        # 权重 (从配置读取)
        self.w_search = float(search_cfg.get('w_search', traj_cfg.get('w_guidance', 0.15)))
        self.w_smoothness = float(traj_cfg.get('w_smoothness', 10.0))
        self.w_safety = float(traj_cfg.get('w_safety', 1.0))
        self.w_acceleration = float(traj_cfg.get('w_acceleration', 1.0))
        
        # 子损失模块
        self.safety_loss = SafetyLoss2D(esdf_maps)
        self.smoothness_loss = SmoothnessLoss2D()
        self.search_loss = SearchLoss()
        
        # 设置不确定性地图
        if uncertainty_maps is not None:
            self.search_loss.set_uncertainty_maps(uncertainty_maps)
        
        # 权重归一化
        self._normalize_weights()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("---------- Search Loss Weights (normalized) ----------")
        print(f"| {'search':<14} = {self.w_search:>8.4f} |")
        print(f"| {'smoothness':<14} = {self.w_smoothness:>8.4f} |")
        print(f"| {'acceleration':<14} = {self.w_acceleration:>8.4f} |")
        print(f"| {'safety':<14} = {self.w_safety:>8.4f} |")
        print("-" * 50)
    
    def _normalize_weights(self):
        """
        根据速度归一化权重
        """
        train_cfg = cfg['training']
        robot_cfg = cfg['robot']
        vel_max_train = float(train_cfg.get('vel_max_train', robot_cfg['max_vel']))
        vel_scale = vel_max_train / 1.0
        
        # 对齐 YOPO_Sim
        self.w_smoothness = self.w_smoothness / (vel_scale ** 5)
        self.w_acceleration = self.w_acceleration / (vel_scale ** 3)
        # search 权重保持不变 (类似 guidance)
    
    def set_esdf_maps(self, esdf_maps: list, map_info: dict = None):
        """设置ESDF地图 (用于安全损失)"""
        self.safety_loss.set_esdf_maps(esdf_maps, map_info)
    
    def set_uncertainty_maps(self, uncertainty_maps: list, map_info: dict = None):
        """设置不确定性地图 (用于搜索损失)"""
        self.search_loss.set_uncertainty_maps(uncertainty_maps, map_info)
    
    def forward(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        map_idx: torch.Tensor
    ) -> tuple:
        """
        计算总损失
        
        Args:
            start_state: [batch, 3, 2] 世界坐标系 [[px, py], [vx, vy], [ax, ay]]
            end_state: [batch, 3, 2] 世界坐标系
            map_idx: [batch] 地图索引
            
        Returns:
            total_cost: [batch]
            costs_dict: 各项损失的字典
        """
        # 安全损失 (避障)
        safety_cost = self.safety_loss(start_state, end_state, map_idx)
        
        # 平滑损失 (Jerk + Acc)
        jerk_cost, acc_cost = self.smoothness_loss(start_state, end_state)
        
        # 搜索损失 (不确定性引导)
        search_cost = self.search_loss(start_state, end_state, map_idx)
        
        # 总损失
        total_cost = (
            self.w_smoothness * jerk_cost +
            self.w_acceleration * acc_cost +
            self.w_safety * safety_cost +
            self.w_search * search_cost
        )
        
        return total_cost, {
            'search': search_cost,
            'safety': safety_cost,
            'smoothness': jerk_cost,
            'acceleration': acc_cost
        }
    
    def forward_with_goal(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        goal: torch.Tensor,
        map_idx: torch.Tensor
    ) -> tuple:
        """
        兼容接口: 与 YopoLoss2D 相同的签名
        
        注意: goal 参数在搜索任务中被忽略, 使用不确定性代替
        """
        return self.forward(start_state, end_state, map_idx)


class HybridSearchLoss(nn.Module):
    """
    混合损失函数 - 同时支持搜索和目标导航
    
    可以在搜索模式和导航模式之间切换:
    - 搜索模式: 使用不确定性引导
    - 导航模式: 使用目标点引导
    """
    
    def __init__(
        self,
        esdf_maps: list = None,
        uncertainty_maps: list = None,
        mode: str = 'search'
    ):
        super().__init__()
        
        traj_cfg = cfg['trajectory']
        search_cfg = cfg.get('search', default={}) if isinstance(cfg._data, dict) else {}
        
        # 公共参数
        self.w_smoothness = float(traj_cfg.get('w_smoothness', 10.0))
        self.w_safety = float(traj_cfg.get('w_safety', 1.0))
        self.w_acceleration = float(traj_cfg.get('w_acceleration', 1.0))
        
        # 模式特定权重
        self.w_search = float(search_cfg.get('w_search', 0.15))
        self.w_guidance = float(traj_cfg.get('w_guidance', 0.15))
        
        # 子损失模块
        self.safety_loss = SafetyLoss2D(esdf_maps)
        self.smoothness_loss = SmoothnessLoss2D()
        self.search_loss = SearchLoss()
        
        # 导入目标引导损失
        from policy.loss import GuidanceLoss2D
        self.guidance_loss = GuidanceLoss2D()
        
        # 设置地图
        if uncertainty_maps is not None:
            self.search_loss.set_uncertainty_maps(uncertainty_maps)
        
        # 模式
        self.mode = mode
        
        # 权重归一化
        self._normalize_weights()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _normalize_weights(self):
        """权重归一化"""
        train_cfg = cfg['training']
        robot_cfg = cfg['robot']
        vel_max_train = float(train_cfg.get('vel_max_train', robot_cfg['max_vel']))
        vel_scale = vel_max_train / 1.0
        
        self.w_smoothness = self.w_smoothness / (vel_scale ** 5)
        self.w_acceleration = self.w_acceleration / (vel_scale ** 3)
    
    def set_mode(self, mode: str):
        """设置模式: 'search' 或 'navigate'"""
        assert mode in ['search', 'navigate'], f"Unknown mode: {mode}"
        self.mode = mode
    
    def set_esdf_maps(self, esdf_maps: list, map_info: dict = None):
        """设置ESDF地图"""
        self.safety_loss.set_esdf_maps(esdf_maps, map_info)
    
    def set_uncertainty_maps(self, uncertainty_maps: list, map_info: dict = None):
        """设置不确定性地图"""
        self.search_loss.set_uncertainty_maps(uncertainty_maps, map_info)
    
    def forward(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        goal: torch.Tensor,
        map_idx: torch.Tensor
    ) -> tuple:
        """
        计算损失
        
        Args:
            start_state: [batch, 3, 2]
            end_state: [batch, 3, 2]
            goal: [batch, 2] (搜索模式下可以是 None 或任意值)
            map_idx: [batch]
        """
        # 公共损失
        safety_cost = self.safety_loss(start_state, end_state, map_idx)
        jerk_cost, acc_cost = self.smoothness_loss(start_state, end_state)
        
        if self.mode == 'search':
            # 搜索模式: 使用不确定性
            task_cost = self.search_loss(start_state, end_state, map_idx)
            task_weight = self.w_search
            task_name = 'search'
        else:
            # 导航模式: 使用目标引导
            task_cost = self.guidance_loss(start_state, end_state, goal)
            task_weight = self.w_guidance
            task_name = 'guidance'
        
        total_cost = (
            self.w_smoothness * jerk_cost +
            self.w_acceleration * acc_cost +
            self.w_safety * safety_cost +
            task_weight * task_cost
        )
        
        return total_cost, {
            task_name: task_cost,
            'safety': safety_cost,
            'smoothness': jerk_cost,
            'acceleration': acc_cost
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Testing YopoSearchLoss...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    H, W = 100, 100
    
    # ESDF 地图 (模拟障碍物)
    esdf_map = np.ones((H, W), dtype=np.float32) * 5.0
    esdf_map[40:60, 40:60] = 0.0  # 障碍物区域
    
    # 不确定性地图
    uncertainty_map = np.ones((H, W), dtype=np.float32)
    uncertainty_map[20:40, 20:40] = 0.1  # 已探索区域
    uncertainty_map[70:90, 70:90] = 1.0  # 高不确定性区域
    
    # 创建损失函数
    loss_fn = YopoSearchLoss(
        esdf_maps=[esdf_map],
        uncertainty_maps=[uncertainty_map]
    )
    loss_fn.set_esdf_maps([esdf_map], {'size': [100, 100], 'resolution': 1.0})
    loss_fn.set_uncertainty_maps([uncertainty_map], {'size': [100, 100], 'resolution': 1.0})
    
    # 测试数据
    batch_size = 4
    start_state = torch.zeros(batch_size, 3, 2, device=device)
    start_state[:, 0, :] = torch.tensor([
        [30, 30], [50, 50], [80, 80], [10, 10]
    ], device=device).float()
    start_state[:, 1, :] = torch.tensor([
        [1, 0], [1, 0], [1, 0], [1, 0]
    ], device=device).float()
    
    end_state = torch.zeros(batch_size, 3, 2, device=device)
    end_state[:, 0, :] = torch.tensor([
        [35, 30], [55, 50], [85, 80], [15, 10]
    ], device=device).float()
    end_state[:, 1, :] = torch.tensor([
        [1, 0], [1, 0], [1, 0], [1, 0]
    ], device=device).float()
    
    map_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # 计算损失
    total_cost, costs_dict = loss_fn(start_state, end_state, map_idx)
    
    print(f"Total cost: {total_cost}")
    print(f"Costs breakdown:")
    for k, v in costs_dict.items():
        print(f"  {k}: {v}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    im = ax.imshow(esdf_map.T, origin='lower', cmap='viridis',
                   extent=[0, 100, 0, 100])
    ax.set_title('ESDF Map (Safety)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1]
    im = ax.imshow(uncertainty_map.T, origin='lower', cmap='hot',
                   extent=[0, 100, 0, 100], vmin=0, vmax=1)
    for i in range(batch_size):
        s = start_state[i, 0].cpu().numpy()
        e = end_state[i, 0].cpu().numpy()
        ax.plot([s[0], e[0]], [s[1], e[1]], 'b-o', linewidth=2)
        ax.annotate(f'{total_cost[i].item():.2f}', (e[0], e[1]), fontsize=10, color='cyan')
    ax.set_title('Uncertainty Map (Search)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('yopo_search_loss_test.png')
    print("Saved to yopo_search_loss_test.png")
    plt.show()
