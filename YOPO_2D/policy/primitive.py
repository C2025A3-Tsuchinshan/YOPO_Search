"""
2D运动基元 (Motion Primitives)
对齐 YOPO_Sim/YOPO/policy/primitive.py 的 2D 版本

关键概念:
- 基元使用极坐标表示: (yaw, radius)
- 网络输出: [delta_yaw, delta_radius, vx, vy, ax, ay] (归一化)
- yaw_diff: 每个 anchor 的 yaw 偏移范围
- radio_range: 规划半径
"""

import numpy as np
import torch
from typing import Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg


class LatticeParam2D:
    """2D基元参数 (对齐 YOPO_Sim LatticeParam)"""
    
    def __init__(self):
        train_cfg = cfg['training']
        traj_cfg = cfg['trajectory']
        robot_cfg = cfg['robot']
        
        # 训练/测试模式区分 (对齐 YOPO_Sim)
        is_train = cfg.get('train', default=True)
        if is_train:
            ratio = 1.0
        else:
            ratio = cfg.get('velocity', default=robot_cfg['max_vel']) / float(train_cfg.get('vel_max_train', robot_cfg['max_vel']))
        
        # 速度/加速度限制
        self.vel_max = ratio * float(train_cfg.get('vel_max_train', robot_cfg['max_vel']))
        self.acc_max = ratio * ratio * float(train_cfg.get('acc_max_train', robot_cfg['max_acc']))
        
        # 轨迹时间 (对齐 YOPO_Sim: sgm_time / ratio)
        self.radio_range = float(traj_cfg.get('radio_range', traj_cfg['planning_horizon'] / 2))
        self.segment_time = traj_cfg['segment_time'] / ratio if not is_train else traj_cfg['segment_time']
        
        # 基元网格参数
        self.horizon_num = int(traj_cfg.get('horizon_num', traj_cfg.get('num_directions', 5)))
        self.vertical_num = 1  # 2D 只有一层
        self.traj_num = self.horizon_num * self.vertical_num
        
        # FOV 参数 (对齐 YOPO_Sim)
        self.horizon_fov = float(traj_cfg.get('direction_fov', 90.0))  # 总覆盖 FOV
        self.horizon_anchor_fov = float(traj_cfg.get('anchor_fov', traj_cfg.get('direction_fov_sim', 30.0)))  # 每个 anchor FOV
        
        # yaw_diff: 网络输出 delta_yaw 的缩放因子 (对齐 YOPO_Sim)
        self.yaw_diff = 0.5 * self.horizon_anchor_fov / 180.0 * np.pi
        
        print("---------- 2D Param (aligned YOPO_Sim) ----------")
        print(f"| {'max speed':<14} = {round(self.vel_max, 2):>8} |")
        print(f"| {'max accel':<14} = {round(self.acc_max, 2):>8} |")
        print(f"| {'traj time':<14} = {round(self.segment_time, 3):>8} |")
        print(f"| {'radio_range':<14} = {round(self.radio_range, 2):>8} |")
        print(f"| {'horizon_num':<14} = {self.horizon_num:>8} |")
        print(f"| {'horizon_fov':<14} = {round(self.horizon_fov, 1):>8}° |")
        print(f"| {'anchor_fov':<14} = {round(self.horizon_anchor_fov, 1):>8}° |")
        print(f"| {'yaw_diff':<14} = {round(np.rad2deg(self.yaw_diff), 2):>8}° |")
        print("-" * 47)


class LatticePrimitive2D(LatticeParam2D):
    """
    2D栅格运动基元 (对齐 YOPO_Sim LatticePrimitive)
    
    网格索引布局 (极坐标，右到左):
        +---+---+---+---+---+
        | 4 | 3 | 2 | 1 | 0 |  (假设 horizon_num=5)
        +---+---+---+---+---+
        -45° -22.5° 0° 22.5° 45°  (假设 horizon_fov=90°)
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 计算基元角度 (对齐 YOPO_Sim: 从右到左, 即从负角度到正角度)
        if self.horizon_num == 1:
            direction_diff = 0.0
        else:
            direction_diff = (self.horizon_fov / 180.0 * np.pi) / self.horizon_num
        
        # 基元位置和角度列表
        lattice_pos_list = []
        lattice_angle_list = []
        
        # 从右到左 (j=0 是最右边，即最负的角度)
        for j in range(self.horizon_num):
            alpha = -direction_diff * (self.horizon_num - 1) / 2 + j * direction_diff
            
            # 位置 = radio_range * [cos(alpha), sin(alpha)]
            pos_node = torch.tensor([
                np.cos(alpha) * self.radio_range,
                np.sin(alpha) * self.radio_range
            ])
            
            lattice_pos_list.append(pos_node)
            lattice_angle_list.append(torch.tensor(alpha))
        
        self.lattice_pos_node = torch.stack(lattice_pos_list).to(dtype=torch.float32, device=self.device)  # [N, 2]
        self.lattice_angle_node = torch.stack(lattice_angle_list).to(dtype=torch.float32, device=self.device)  # [N]
        
        # 兼容旧接口
        self.num_directions = self.traj_num
        self.angles = self.lattice_angle_node.cpu().numpy()
        self.angles_tensor = self.lattice_angle_node
        self.end_positions = self.lattice_pos_node.cpu().numpy()
        self.end_positions_tensor = self.lattice_pos_node
        self.planning_horizon = 2 * self.radio_range
    
    def getStateLattice(self, id=None):
        """获取基元位置 (对齐 YOPO_Sim)"""
        if id is not None:
            return self.lattice_pos_node[id, :]
        else:
            return self.lattice_pos_node
    
    def getAngleLattice(self, id=None):
        """获取基元角度 (对齐 YOPO_Sim)"""
        if id is not None:
            return self.lattice_angle_node[id]
        else:
            return self.lattice_angle_node
    
    def convert_ImageGrid_LatticeID(self, id):
        """图像网格到基元ID的转换 (对齐 YOPO_Sim: 顺序相反)"""
        return self.traj_num - id - 1
    
    def get_primitive_endpoints(self, batch_size: int = 1) -> torch.Tensor:
        """获取所有基元的终点位置 (机体坐标系)"""
        return self.end_positions_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    
    def get_primitive_angles(self) -> torch.Tensor:
        """获取所有基元的方向角度"""
        return self.angles_tensor


class StateTransform2D:
    """
    2D状态变换 (对齐 YOPO_Sim StateTransform)
    
    关键方法:
    - normalize_obs: 归一化状态 [vel/vel_max, acc/acc_max, goal/goal_length]
    - prepare_input: 将机体系状态变换到各基元坐标系
    - pred_to_endstate: 将网络输出转换为机体系终止状态
    """
    
    def __init__(self):
        self.lattice_primitive = LatticePrimitive2D.get_instance()
        self.goal_length = float(cfg['training'].get('goal_length', 2.0 * self.lattice_primitive.radio_range))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"StateTransform2D: goal_length={self.goal_length:.2f}, vel_max={self.lattice_primitive.vel_max:.2f}, acc_max={self.lattice_primitive.acc_max:.2f}")
    
    # 兼容旧接口
    @property
    def max_vel(self):
        return self.lattice_primitive.vel_max
    
    @property
    def max_acc(self):
        return self.lattice_primitive.acc_max
    
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        归一化状态 (对齐 YOPO_Sim normalize_obs)
        
        Args:
            state: [batch_size, 6] = [vx, vy, ax, ay, goal_x, goal_y] (机体坐标系)
            
        Returns:
            normalized: [batch_size, 6]
        """
        vel = state[:, :2] / self.lattice_primitive.vel_max
        acc = state[:, 2:4] / self.lattice_primitive.acc_max
        goal = state[:, 4:6]
        
        # 目标归一化: clamp to unit length (对齐 YOPO_Sim)
        goal_norm = torch.norm(goal, dim=-1, keepdim=True)
        goal = goal / goal_norm.clamp(min=self.goal_length)
        
        return torch.cat([vel, acc, goal], dim=-1)

    def prepare_input(self, obs: torch.Tensor) -> torch.Tensor:
        """
        将归一化状态变换到各基元坐标系 (对齐 YOPO_Sim prepare_input)
        
        Args:
            obs: [batch_size, 6] 归一化的机体系状态 [vx, vy, ax, ay, gx, gy]
            
        Returns:
            state_prim: [batch_size, num_primitives, 6] 各基元坐标系下的状态
            
        Note:
            使用 .flip(0) 是为了对齐 YOPO_Sim 的训练逻辑：
            - 网络输出的 grid 索引是从左到右 (primitive[0] = 图像最右边 = 网络的最高索引)
            - 基元存储是从右到左 (lattice[0] = -36°)
            - flip 将 lattice 顺序反转，使 angles[0] 对应 primitive[0]
        """
        B = obs.shape[0]
        N = self.lattice_primitive.traj_num
        
        # 获取所有基元角度并倒序 (对齐 YOPO_Sim: lattice 和 grid 顺序相反)
        angles = self.lattice_primitive.getAngleLattice().flip(0).to(obs.device)  # [N]
        
        # 构建旋转矩阵 (从机体系到基元系: 逆时针旋转 -alpha)
        cos_a = torch.cos(-angles)  # [N]
        sin_a = torch.sin(-angles)  # [N]
        
        # 分离 vel, acc, goal
        vel = obs[:, :2]  # [B, 2]
        acc = obs[:, 2:4]  # [B, 2]
        goal = obs[:, 4:6]  # [B, 2]
        
        def rotate_to_primitive(vec: torch.Tensor) -> torch.Tensor:
            """将向量从机体系旋转到基元系"""
            # vec: [B, 2], angles: [N]
            # output: [B, N, 2]
            v = vec.unsqueeze(1)  # [B, 1, 2]
            cos = cos_a.view(1, -1, 1)  # [1, N, 1]
            sin = sin_a.view(1, -1, 1)  # [1, N, 1]
            x = cos * v[..., 0:1] - sin * v[..., 1:2]
            y = sin * v[..., 0:1] + cos * v[..., 1:2]
            return torch.cat([x, y], dim=-1)  # [B, N, 2]
        
        vel_p = rotate_to_primitive(vel)  # [B, N, 2]
        acc_p = rotate_to_primitive(acc)  # [B, N, 2]
        goal_p = rotate_to_primitive(goal)  # [B, N, 2]
        
        return torch.cat([vel_p, acc_p, goal_p], dim=-1)  # [B, N, 6]
    
    def denormalize_endstate(self, endstate_pred: torch.Tensor) -> torch.Tensor:
        """
        将网络输出转换为机体系终止状态 (对齐 YOPO_Sim pred_to_endstate)
        
        Args:
            endstate_pred: [batch_size, num_primitives, 6] 网络输出 (tanh 激活后)
                          = [delta_yaw, delta_radius, vx, vy, ax, ay] (归一化)
                          其中:
                          - delta_yaw: [-1, 1] -> [-yaw_diff, +yaw_diff]
                          - delta_radius: [-1, 1] -> [0, 2*radio_range]
                          - vx, vy, ax, ay: [-1, 1] -> [-max, +max]
                          
        Returns:
            endstate: [batch_size, num_primitives, 6] 机体系终止状态
                     = [px, py, vx, vy, ax, ay]
                     
        Note:
            使用 .flip(0) 对齐 YOPO_Sim pred_to_endstate 的逻辑：
            - 网络输出索引 i 对应 flip 后的 angles[i]
            - 这样 primitive[0] 实际对应最大角度 (+36°)
            - primitive[N-1] 对应最小角度 (-36°)
        """
        B = endstate_pred.shape[0]
        N = self.lattice_primitive.traj_num
        
        # 获取基元角度并倒序 (对齐 YOPO_Sim)
        yaw_anchors = self.lattice_primitive.getAngleLattice().flip(0).to(endstate_pred.device)  # [N]
        yaw_anchors = yaw_anchors.view(1, -1)  # [1, N]
        
        # 解析网络输出
        delta_yaw = endstate_pred[:, :, 0] * self.lattice_primitive.yaw_diff  # [B, N]
        delta_radius = (endstate_pred[:, :, 1] + 1.0) * self.lattice_primitive.radio_range  # [B, N], 范围 [0, 2*radio_range]
        
        # 计算终止位置 (基元系下的极坐标 -> 机体系)
        # 终止点的 yaw = anchor_yaw + delta_yaw
        total_yaw = yaw_anchors + delta_yaw  # [B, N]
        
        # 位置 (机体系)
        end_x = torch.cos(total_yaw) * delta_radius  # [B, N]
        end_y = torch.sin(total_yaw) * delta_radius  # [B, N]
        end_pos = torch.stack([end_x, end_y], dim=-1)  # [B, N, 2]
        
        # 速度和加速度 (基元系 -> 机体系)
        vel_p = endstate_pred[:, :, 2:4] * self.lattice_primitive.vel_max  # [B, N, 2]
        acc_p = endstate_pred[:, :, 4:6] * self.lattice_primitive.acc_max  # [B, N, 2]
        
        # 从基元系旋转到机体系
        cos_a = torch.cos(yaw_anchors).unsqueeze(-1)  # [1, N, 1]
        sin_a = torch.sin(yaw_anchors).unsqueeze(-1)  # [1, N, 1]
        
        def rotate_to_body(vec: torch.Tensor) -> torch.Tensor:
            """将向量从基元系旋转到机体系"""
            x = cos_a * vec[..., 0:1] - sin_a * vec[..., 1:2]
            y = sin_a * vec[..., 0:1] + cos_a * vec[..., 1:2]
            return torch.cat([x, y], dim=-1)
        
        vel_b = rotate_to_body(vel_p)  # [B, N, 2]
        acc_b = rotate_to_body(acc_p)  # [B, N, 2]
        
        return torch.cat([end_pos, vel_b, acc_b], dim=-1)  # [B, N, 6]
    
    def body_to_world(
        self,
        state_body: torch.Tensor,
        position: torch.Tensor,
        heading: torch.Tensor
    ) -> torch.Tensor:
        """机体坐标系转世界坐标系"""
        squeeze = False
        if state_body.dim() == 2:
            state_body = state_body.unsqueeze(1)
            squeeze = True
        
        batch_size, N, _ = state_body.shape
        
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        R = torch.stack([
            torch.stack([cos_h, -sin_h], dim=-1),
            torch.stack([sin_h, cos_h], dim=-1)
        ], dim=-2)  # [batch_size, 2, 2]
        
        pos_body = state_body[:, :, :2]
        vel_body = state_body[:, :, 2:4]
        acc_body = state_body[:, :, 4:6]
        
        pos_world = torch.einsum('bij,bnj->bni', R, pos_body) + position.unsqueeze(1)
        vel_world = torch.einsum('bij,bnj->bni', R, vel_body)
        acc_world = torch.einsum('bij,bnj->bni', R, acc_body)
        
        state_world = torch.cat([pos_world, vel_world, acc_world], dim=-1)
        
        if squeeze:
            state_world = state_world.squeeze(1)
        
        return state_world
    
    def world_to_body(
        self,
        state_world: torch.Tensor,
        position: torch.Tensor,
        heading: torch.Tensor
    ) -> torch.Tensor:
        """世界坐标系转机体坐标系"""
        squeeze = False
        if state_world.dim() == 2:
            state_world = state_world.unsqueeze(1)
            squeeze = True
        
        batch_size, N, _ = state_world.shape
        
        cos_h = torch.cos(-heading)
        sin_h = torch.sin(-heading)
        R_inv = torch.stack([
            torch.stack([cos_h, -sin_h], dim=-1),
            torch.stack([sin_h, cos_h], dim=-1)
        ], dim=-2)
        
        pos_world = state_world[:, :, :2]
        vel_world = state_world[:, :, 2:4]
        acc_world = state_world[:, :, 4:6]
        
        pos_body = torch.einsum('bij,bnj->bni', R_inv, pos_world - position.unsqueeze(1))
        vel_body = torch.einsum('bij,bnj->bni', R_inv, vel_world)
        acc_body = torch.einsum('bij,bnj->bni', R_inv, acc_world)
        
        state_body = torch.cat([pos_body, vel_body, acc_body], dim=-1)
        
        if squeeze:
            state_body = state_body.squeeze(1)
        
        return state_body


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 测试运动基元
    primitives = LatticePrimitive2D()
    
    # 可视化基元
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 机器人位置
    robot_pos = np.array([0.0, 0.0])
    robot_heading = 0.0
    
    # 绘制机器人
    ax.plot(robot_pos[0], robot_pos[1], 'go', markersize=15, label='Robot')
    ax.arrow(robot_pos[0], robot_pos[1], 
             np.cos(robot_heading) * 0.5, np.sin(robot_heading) * 0.5,
             head_width=0.2, head_length=0.1, fc='g', ec='g')
    
    # 绘制基元终点
    colors = plt.cm.rainbow(np.linspace(0, 1, primitives.num_directions))
    for i, (pos, angle) in enumerate(zip(primitives.end_positions, primitives.angles)):
        ax.plot([robot_pos[0], pos[0]], [robot_pos[1], pos[1]], 
                '-', color=colors[i], linewidth=2, alpha=0.7)
        ax.plot(pos[0], pos[1], 'o', color=colors[i], markersize=10)
        ax.annotate(f'{np.rad2deg(angle):.0f}°', (pos[0], pos[1]), 
                   fontsize=8, ha='center', va='bottom')
    
    ax.set_xlim(-1, primitives.planning_horizon + 1)
    ax.set_ylim(-primitives.planning_horizon/2 - 1, primitives.planning_horizon/2 + 1)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('2D Motion Primitives')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('primitives_test.png')
    plt.show()
