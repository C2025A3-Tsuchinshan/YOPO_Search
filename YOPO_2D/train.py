"""
2D YOPO 训练器
"""

import os
import time
import atexit
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from policy.network import YopoNetwork2D
from policy.loss import YopoLoss2D
from policy.primitive import LatticePrimitive2D, StateTransform2D
from dataset import YopoDataset2D


class YopoTrainer2D:
    """2D YOPO训练器"""
    
    def __init__(
        self,
        learning_rate: float = None,
        batch_size: int = None,
        checkpoint_path: str = None,
        log_dir: str = None,
        save_on_exit: bool = True,
        data_dir: str = None,   
    ):
        train_cfg = cfg['training']
        
        self.learning_rate = float(learning_rate or train_cfg['learning_rate'])
        self.batch_size = batch_size or train_cfg['batch_size']
        self.save_interval = train_cfg['save_interval']
        self.max_grad_norm = 0.1  # 对齐 YOPO_Sim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 优化 CUDA 性能
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # 网络
        print("Loading network...")
        self.network = YopoNetwork2D().to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.network.load_state_dict(state_dict)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("Training from scratch")
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # 学习率调度器 (可选，YOPO_Sim 没有使用)
        self.use_scheduler = cfg['training'].get('use_scheduler', False)
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg['training']['epochs'],
                eta_min=1e-6
            )
        else:
            self.scheduler = None
        
        # 损失函数
        self.loss_fn = YopoLoss2D()
        
        # 状态变换
        self.state_transform = StateTransform2D()
        self.primitives = LatticePrimitive2D.get_instance()
        
        # 日志
        self.log_dir = self._get_log_dir(log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"Logging to {self.log_dir}")
        
        # 保存
        if save_on_exit:
            self._exit_func = atexit.register(self.save_model)
        
        self.data_dir = data_dir
        print(f"data_dir: {self.data_dir}")
        
        self.epoch = 0
    
    def _get_log_dir(self, base_dir: str = None) -> str:
        """获取日志目录"""
        base_dir = base_dir or os.path.join(os.path.dirname(__file__), 'saved')
        os.makedirs(base_dir, exist_ok=True)
        
        # 查找下一个编号
        existing = [d for d in os.listdir(base_dir) if d.startswith('YOPO2D_')]
        nums = [int(d.split('_')[1]) for d in existing if d.split('_')[1].isdigit()]
        next_num = max(nums, default=-1) + 1
        
        log_dir = os.path.join(base_dir, f'YOPO2D_{next_num}')
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    def setup_data(
        self,
        data_dir: str = None,
        num_maps: int = 10,
        samples_per_map: int = 5000
    ):
        """设置数据集"""
        print("Loading dataset...")
        
        self.train_dataset = YopoDataset2D(
            data_dir=data_dir,
            mode='train',
            num_maps=num_maps,
            samples_per_map=samples_per_map
        )
        
        self.val_dataset = YopoDataset2D(
            data_dir=data_dir,
            mode='valid',
            num_maps=num_maps,
            samples_per_map=samples_per_map
        )
        self.val_dataset.esdf_maps = self.train_dataset.esdf_maps
        self.val_dataset.indices = self.val_dataset.indices  # 确保使用正确的索引
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        
        # 设置ESDF到损失函数
        self.loss_fn.set_esdf_maps(
            self.train_dataset.esdf_maps,
            self.train_dataset.map_info
        )
        
        print(f"Train: {len(self.train_dataset)} samples, Val: {len(self.val_dataset)} samples")
    
    def train(self, epochs: int = None):
        """训练"""
        epochs = epochs or cfg['training']['epochs']
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 学习率调度 (可选)
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 日志
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.learning_rate
            self.writer.add_scalar('LR', current_lr, epoch)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # 保存
            if (epoch + 1) % self.save_interval == 0:
                self.save_model(f'epoch{epoch + 1}.pth')
        
        print("Training complete!")
        self.save_model('final.pth')
    
    def train_epoch(self) -> float:
        """训练一个epoch (对齐 YOPO_Sim 的日志格式)"""
        self.network.train()
        
        total_loss = 0.0
        num_batches = 0
        traj_losses, score_losses = [], []
        smooth_losses, safety_losses, goal_losses, acc_losses = [], [], [], []
        
        inspect_interval = max(1, len(self.train_loader) // 16)  # 对齐 YOPO_Sim
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for step, batch in enumerate(pbar):
            lidar, position, heading, velocity, acceleration, goal, map_idx = [
                b.to(self.device, non_blocking=True) for b in batch
            ]
            
            # 构建机体状态
            goal_body = self._world_to_body_batch(goal - position, heading)
            state_body = torch.cat([velocity, acceleration, goal_body], dim=-1)
            
            # 前向传播
            self.optimizer.zero_grad()
            endstate_pred, score_pred = self.network.inference(lidar, state_body)
            
            # 计算损失
            loss, loss_dict = self._compute_loss(
                endstate_pred, score_pred,
                position, heading, velocity, acceleration, goal, map_idx
            )
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 记录详细损失
            traj_losses.append(loss_dict['traj'])
            score_losses.append(loss_dict['score'])
            smooth_losses.append(loss_dict.get('smooth', 0))
            safety_losses.append(loss_dict.get('safety', 0))
            goal_losses.append(loss_dict.get('goal', 0))
            acc_losses.append(loss_dict.get('acc', 0))
            
            # 按 step 记录到 TensorBoard (对齐 YOPO_Sim)
            if step % inspect_interval == inspect_interval - 1:
                global_step = self.epoch * len(self.train_loader) + step
                self.writer.add_scalar('Train/TrajLoss', np.mean(traj_losses), global_step)
                self.writer.add_scalar('Train/ScoreLoss', np.mean(score_losses), global_step)
                self.writer.add_scalar('Detail/SmoothLoss', np.mean(smooth_losses), global_step)
                self.writer.add_scalar('Detail/SafetyLoss', np.mean(safety_losses), global_step)
                self.writer.add_scalar('Detail/GoalLoss', np.mean(goal_losses), global_step)
                self.writer.add_scalar('Detail/AccelLoss', np.mean(acc_losses), global_step)
                # 清空列表
                traj_losses, score_losses = [], []
                smooth_losses, safety_losses, goal_losses, acc_losses = [], [], [], []
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """验证 (对齐 YOPO_Sim 的日志格式)"""
        self.network.eval()
        
        total_loss = 0.0
        num_batches = 0
        traj_losses, score_losses = [], []
        
        for batch in self.val_loader:
            lidar, position, heading, velocity, acceleration, goal, map_idx = [
                b.to(self.device, non_blocking=True) for b in batch
            ]
            
            goal_body = self._world_to_body_batch(goal - position, heading)
            state_body = torch.cat([velocity, acceleration, goal_body], dim=-1)
            
            endstate_pred, score_pred = self.network.inference(lidar, state_body)
            
            loss, loss_dict = self._compute_loss(
                endstate_pred, score_pred,
                position, heading, velocity, acceleration, goal, map_idx
            )
            
            total_loss += loss.item()
            num_batches += 1
            traj_losses.append(loss_dict['traj'])
            score_losses.append(loss_dict['score'])
        
        # 记录验证损失 (按 epoch)
        self.writer.add_scalar('Eval/TrajLoss', np.mean(traj_losses), self.epoch)
        self.writer.add_scalar('Eval/ScoreLoss', np.mean(score_losses), self.epoch)
        
        return total_loss / num_batches
    
    def _compute_loss(
        self,
        endstate_pred: torch.Tensor,
        score_pred: torch.Tensor,
        position: torch.Tensor,
        heading: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        goal: torch.Tensor,
        map_idx: torch.Tensor
    ) -> tuple:
        """计算损失"""
        batch_size = position.shape[0]
        num_primitives = endstate_pred.shape[1]
        loss_weight = cfg['training'].get('loss_weight', [1.0, 1.0])
        traj_w = float(loss_weight[0]) if len(loss_weight) > 0 else 1.0
        score_w = float(loss_weight[1]) if len(loss_weight) > 1 else 1.0
        
        # 起始状态 (世界系)
        vel_world = self._body_to_world_batch(velocity, heading)
        acc_world = self._body_to_world_batch(acceleration, heading)
        start_state = torch.stack([position, vel_world, acc_world], dim=1)  # [B, 3, 2]
        
        # 终止状态 (世界系)
        # endstate_pred: [B, N, 6] = [dx, dy, dvx, dvy, dax, day] in body frame
        end_pos_body = endstate_pred[:, :, :2]
        end_vel_body = endstate_pred[:, :, 2:4]
        end_acc_body = endstate_pred[:, :, 4:6]
        
        end_pos_world = self._body_to_world_batch(
            end_pos_body.reshape(-1, 2),
            heading.repeat_interleave(num_primitives)
        ).reshape(batch_size, num_primitives, 2) + position.unsqueeze(1)
        
        end_vel_world = self._body_to_world_batch(
            end_vel_body.reshape(-1, 2),
            heading.repeat_interleave(num_primitives)
        ).reshape(batch_size, num_primitives, 2)
        
        end_acc_world = self._body_to_world_batch(
            end_acc_body.reshape(-1, 2),
            heading.repeat_interleave(num_primitives)
        ).reshape(batch_size, num_primitives, 2)
        
        # 展平计算损失
        start_state_flat = start_state.unsqueeze(1).expand(-1, num_primitives, -1, -1)
        start_state_flat = start_state_flat.reshape(-1, 3, 2)
        
        end_state_flat = torch.stack([end_pos_world, end_vel_world, end_acc_world], dim=2)
        end_state_flat = end_state_flat.reshape(-1, 3, 2)
        
        goal_flat = goal.unsqueeze(1).expand(-1, num_primitives, -1).reshape(-1, 2)
        map_idx_flat = map_idx.unsqueeze(1).expand(-1, num_primitives).reshape(-1)
        
        # 计算轨迹损失
        traj_cost, costs_dict = self.loss_fn(
            start_state_flat, end_state_flat, goal_flat, map_idx_flat
        )
        
        traj_cost = traj_cost.reshape(batch_size, num_primitives)
        traj_loss = traj_cost.mean()
        
        # Score loss: 预测得分应该接近实际代价
        score_label = traj_cost.detach()
        score_loss = F.smooth_l1_loss(score_pred, score_label)
        
        # 总损失
        total_loss = traj_w * traj_loss + score_w * score_loss
        
        # 详细损失 (对齐 YOPO_Sim)
        return total_loss, {
            'traj': traj_loss.item(),
            'score': score_loss.item(),
            'smooth': costs_dict['smoothness'].mean().item() if 'smoothness' in costs_dict else 0,
            'safety': costs_dict['safety'].mean().item() if 'safety' in costs_dict else 0,
            'goal': costs_dict['guidance'].mean().item() if 'guidance' in costs_dict else 0,
            'acc': costs_dict['acceleration'].mean().item() if 'acceleration' in costs_dict else 0,
        }
    
    def _body_to_world_batch(self, vec_body: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        """批量机体到世界坐标变换"""
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        
        vec_world = torch.stack([
            cos_h * vec_body[:, 0] - sin_h * vec_body[:, 1],
            sin_h * vec_body[:, 0] + cos_h * vec_body[:, 1]
        ], dim=-1)
        
        return vec_world
    
    def _world_to_body_batch(self, vec_world: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        """批量世界到机体坐标变换"""
        cos_h = torch.cos(-heading)
        sin_h = torch.sin(-heading)
        
        vec_body = torch.stack([
            cos_h * vec_world[:, 0] - sin_h * vec_world[:, 1],
            sin_h * vec_world[:, 0] + cos_h * vec_world[:, 1]
        ], dim=-1)
        
        return vec_body
    
    def save_model(self, filename: str = None):
        """保存模型"""
        if filename is None:
            filename = f'epoch{self.epoch + 1}.pth'
        
        path = os.path.join(self.log_dir, filename)
        torch.save(self.network.state_dict(), path)
        print(f"Model saved to {path}")


def train():
    """训练入口"""
    trainer = YopoTrainer2D(data_dir='./dataset/dataset2')
    trainer.setup_data(num_maps=10, samples_per_map=5000, data_dir=trainer.data_dir)
    trainer.train(epochs=cfg['training']['epochs'])


if __name__ == "__main__":
    train()
