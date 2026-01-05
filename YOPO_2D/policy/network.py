"""
2D YOPO 神经网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from policy.primitive import LatticePrimitive2D, StateTransform2D


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, activation=nn.ReLU):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(out + x)


class ResidualConv1DBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        return self.act(out)


class ResNet1DBackbone(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layers=(2, 2, 2, 2), base_channels: int = 64):
        super().__init__()
        self.in_channels = base_channels

        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU()
        )

        self.layer1 = self._make_layer(base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=1)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 8, output_dim)

    def _make_layer(self, out_channels: int, blocks: int, stride: int):
        layers = [BasicBlock1D(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        return self.fc(x)


class Conv1DBackbone(nn.Module):
    """1D?????? (??????)"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.stage1 = nn.Sequential(
            ResidualConv1DBlock(64),
            ResidualConv1DBlock(64)
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.stage2 = nn.Sequential(
            ResidualConv1DBlock(128),
            ResidualConv1DBlock(128)
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.stage3 = nn.Sequential(
            ResidualConv1DBlock(256)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        # x: [batch, num_beams]
        x = x.unsqueeze(1)  # [batch, 1, num_beams]
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = x.squeeze(-1)  # [batch, 256]
        x = self.fc(x)
        return x


class MLPBackbone(nn.Module):
    """MLP????"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(
            ResidualMLPBlock(256),
            ResidualMLPBlock(256),
            ResidualMLPBlock(256)
        )
        self.output_proj = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)


class YopoHead2D(nn.Module):
    """YOPO输出头"""
    
    def __init__(self, input_dim: int, num_primitives: int, output_per_primitive: int = 7):
        """
        Args:
            input_dim: 输入特征维度
            num_primitives: 基元数量
            output_per_primitive: 每个基元输出 [dx, dy, dvx, dvy, dax, day, score]
        """
        super().__init__()
        
        self.output_per_primitive = output_per_primitive
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_per_primitive)
        )
    
    def forward(self, x):
        # x: [batch, num_primitives, input_dim]
        orig_shape = x.shape[:-1]
        out = self.net(x.view(-1, x.shape[-1]))
        return out.view(*orig_shape, self.output_per_primitive)


class YopoNetwork2D(nn.Module):
    """
    2D YOPO网络 (对齐 YOPO_Sim YopoNetwork)
    
    结构:
    - lidar_backbone: 处理激光数据 -> hidden_dim 特征
    - state_encoder: 处理状态 -> hidden_dim/2 特征 (每个基元)
    - head: 融合特征 -> [dyaw, dr, vx, vy, ax, ay, score] 输出
    """
    
    def __init__(self):
        super().__init__()
        
        net_cfg = cfg['network']
        traj_cfg = cfg['trajectory']
        
        self.input_dim = net_cfg['input_dim']  # 激光束数量
        self.state_dim = net_cfg['state_dim']  # [vx, vy, ax, ay, goal_x, goal_y]
        self.hidden_dim = net_cfg['hidden_dim']
        
        # 基元数量 (从配置或自动计算)
        self.num_primitives = int(traj_cfg.get('num_primitives', traj_cfg['horizon_num'] * traj_cfg.get('vertical_num', 1)))
        
        # 状态变换
        self.state_transform = StateTransform2D()
        
        # 骨干网络 (对齐 YOPO_Sim YopoBackbone)
        backbone_type = net_cfg.get('backbone', 'resnet1d')
        if backbone_type == 'cnn1d':
            self.lidar_backbone = Conv1DBackbone(self.input_dim, self.hidden_dim)
            print(f"YopoNetwork2D (Conv1D) params: {sum(p.numel() for p in self.lidar_backbone.parameters()):,}")
        elif backbone_type == 'resnet1d':
            self.lidar_backbone = ResNet1DBackbone(self.input_dim, self.hidden_dim)
            print(f"YopoNetwork2D (ResNet1D) params: {sum(p.numel() for p in self.lidar_backbone.parameters()):,}")
        else:
            self.lidar_backbone = MLPBackbone(self.input_dim, self.hidden_dim)
            print(f"YopoNetwork2D (MLP) params: {sum(p.numel() for p in self.lidar_backbone.parameters()):,}")
        
        # 状态编码器 (对齐 YOPO_Sim: 状态直接拼接，不使用额外编码器)
        # 但为了2D版本的表现，保留一个简单编码器
        self.state_encoder = nn.Identity()  # 对齐 YOPO_Sim state_backbone = nn.Sequential()
        
        # 输出头 (对齐 YOPO_Sim YopoHead)
        combined_dim = self.hidden_dim + self.state_dim
        self.head = YopoHead2D(combined_dim, self.num_primitives, output_per_primitive=7)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"YopoNetwork2D: num_primitives={self.num_primitives}, hidden_dim={self.hidden_dim}")
    
    def forward(self, lidar: torch.Tensor, state: torch.Tensor) -> tuple:
        """
        前向传播 (对齐 YOPO_Sim YopoNetwork.forward)
        
        Args:
            lidar: [batch, num_beams] 归一化的激光数据
            state: [batch, num_primitives, 6] 各基元坐标系下的归一化状态
            
        Returns:
            endstate: [batch, num_primitives, 6] 终止状态偏移 (tanh激活)
            score: [batch, num_primitives] 轨迹评分 (softplus激活)
        """
        batch_size = lidar.shape[0]
        
        # 提取特征
        lidar_feat = self.lidar_backbone(lidar)  # [batch, hidden_dim]
        
        # 状态特征: 直接使用 (对齐 YOPO_Sim: state_backbone = nn.Sequential())
        state_feat = self.state_encoder(state)  # [batch, num_primitives, 6]

        # 融合特征 (按 primitive)
        lidar_feat = lidar_feat.unsqueeze(1).expand(-1, self.num_primitives, -1)  # [batch, num_primitives, hidden_dim]
        combined = torch.cat([state_feat, lidar_feat], dim=-1)  # [batch, num_primitives, hidden_dim + 6]
        
        # 输出
        output = self.head(combined)  # [batch, num_primitives, 7]
        # per-primitive: [dyaw, radius, vx, vy, ax, ay, score]
        
        # 分离终止状态和评分 (对齐 YOPO_Sim)
        endstate = torch.tanh(output[:, :, :6])  # [batch, num_primitives, 6]
        score = F.softplus(output[:, :, 6])  # [batch, num_primitives]
        
        return endstate, score
    
    def inference(self, lidar: torch.Tensor, state: torch.Tensor) -> tuple:
        """
        推理模式 (对齐 YOPO_Sim YopoNetwork.inference)
        
        流程:
        1. 归一化输入状态
        2. 变换到各基元坐标系
        3. 前向传播
        4. 将预测转换为机体系终止状态
        
        Args:
            lidar: [batch, num_beams] 归一化的激光数据
            state: [batch, 6] 原始状态 (机体坐标系) [vx, vy, ax, ay, gx, gy]
            
        Returns:
            endstate: [batch, num_primitives, 6] 终止状态 (机体坐标系) [px, py, vx, vy, ax, ay]
            score: [batch, num_primitives]
        """
        # 1. 归一化状态 (对齐 YOPO_Sim normalize_obs)
        state_normalized = self.state_transform.normalize_state(state)
        
        # 2. 变换到各基元坐标系 (对齐 YOPO_Sim prepare_input)
        state_prim = self.state_transform.prepare_input(state_normalized)  # [batch, num_primitives, 6]
        
        # 3. 前向传播
        endstate_pred, score = self.forward(lidar, state_prim)
        
        # 4. 反归一化终止状态 (对齐 YOPO_Sim pred_to_endstate)
        endstate = self.state_transform.denormalize_endstate(endstate_pred)
        
        return endstate, score
    
    def select_best_trajectory(
        self,
        endstate: torch.Tensor,
        score: torch.Tensor
    ) -> tuple:
        """
        选择最优轨迹
        
        Args:
            endstate: [batch, num_primitives, 6]
            score: [batch, num_primitives]
            
        Returns:
            best_endstate: [batch, 6]
            best_idx: [batch]
        """
        # 选择得分最低的轨迹
        best_idx = torch.argmin(score, dim=-1)  # [batch]
        
        batch_idx = torch.arange(endstate.shape[0], device=endstate.device)
        best_endstate = endstate[batch_idx, best_idx]  # [batch, 6]
        
        return best_endstate, best_idx


if __name__ == "__main__":
    # 测试网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    net = YopoNetwork2D().to(device)
    print(net)
    
    # 测试输入
    batch_size = 4
    lidar = torch.randn(batch_size, cfg['network']['input_dim']).to(device)
    state = torch.randn(batch_size, cfg['network']['state_dim']).to(device)
    
    # 前向传播
    endstate, score = net(lidar, state)
    print(f"Endstate shape: {endstate.shape}")
    print(f"Score shape: {score.shape}")
    
    # 推理
    net.eval()
    with torch.no_grad():
        endstate, score = net.inference(lidar, state)
        best_endstate, best_idx = net.select_best_trajectory(endstate, score)
    
    print(f"Best endstate shape: {best_endstate.shape}")
    print(f"Best idx: {best_idx}")
    
    # 参数数量
    num_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {num_params:,}")
