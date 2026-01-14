"""
搜索任务测试脚本 (Search Task Test)

单机搜索单目标场景的仿真与可视化。
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
from matplotlib.collections import LineCollection
import argparse
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from simulator.map_generator import Map2D
from simulator.sensor import Lidar2D
from simulator.dynamics import Robot2D, Poly5Solver2D
from simulator.uncertainty_map import UncertaintyMap
from policy.network import YopoNetwork2D
from policy.primitive import LatticePrimitive2D


class SearchSimulator:
    """搜索任务仿真器"""
    
    def __init__(
        self,
        model_path: str = None,
        use_gpu: bool = True
    ):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 加载网络
        self.network = YopoNetwork2D().to(self.device)
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.network.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: Using untrained model!")
        self.network.eval()
        
        # 组件
        self.map_2d = None
        self.uncertainty_map = None
        self.robot = Robot2D()
        self.lidar = Lidar2D()
        self.primitives = LatticePrimitive2D.get_instance()
        
        # 参数
        sim_cfg = cfg['simulation']
        search_cfg = cfg.get('search', default={}) if isinstance(cfg._data, dict) else {}
        
        self.dt = sim_cfg['dt']
        self.max_steps = sim_cfg['max_steps']
        self.target_radius = float(search_cfg.get('target_radius', 1.0))
        self.sensor_radius = float(search_cfg.get('sensor_radius', 5.0))
        
        # 状态
        self.target_position = None
        self.trajectory = []
        self.found_target = False
        self.step_count = 0
    
    def reset(
        self,
        map_type: str = None,
        start_pos: np.ndarray = None,
        target_pos: np.ndarray = None,
        seed: int = None
    ):
        """重置仿真"""
        seed = seed or np.random.randint(0, 10000)
        np.random.seed(seed)
        
        # 生成地图
        self.map_2d = Map2D(seed=seed)
        self.map_2d.generate(map_type)
        
        # 创建不确定性地图
        search_cfg = cfg.get('search', default={}) if isinstance(cfg._data, dict) else {}
        self.uncertainty_map = UncertaintyMap(
            map_size=cfg['env']['map_size'],
            resolution=cfg['env']['resolution'],
            initial_prob=float(search_cfg.get('initial_prob', 0.5)),
            detection_prob=float(search_cfg.get('detection_prob', 0.9)),
            false_alarm_prob=float(search_cfg.get('false_alarm_prob', 0.1)),
            k_eta=float(search_cfg.get('k_eta', 1.0)),
            sensor_radius=self.sensor_radius
        )
        
        # 设置起点
        if start_pos is None:
            start_pos = self.map_2d.sample_free_position(self.robot.radius)
        
        # 设置目标
        if target_pos is None:
            for _ in range(100):
                target_pos = self.map_2d.sample_free_position(self.robot.radius)
                if target_pos is not None and np.linalg.norm(target_pos - start_pos) > 20:
                    break
        
        self.target_position = target_pos
        print(f"Target position: {self.target_position}")
        
        # 重置机器人
        heading = np.random.uniform(-np.pi, np.pi)
        self.robot.reset(
            position=start_pos,
            velocity=np.array([0.5, 0.0]),
            heading=heading
        )
        
        # 清空状态
        self.trajectory = [start_pos.copy()]
        self.found_target = False
        self.step_count = 0
        
        return self.get_observation()
    
    def get_observation(self) -> dict:
        """获取观测"""
        state = self.robot.get_state()
        
        # 激光扫描
        lidar_scan = self.lidar.scan(
            self.map_2d, state['position'], state['heading'], add_noise=False
        )
        lidar_normalized = self.lidar.normalize_ranges(lidar_scan)
        
        # 机体状态 (搜索任务不需要目标信息，使用随机方向)
        vel_body = self.robot.velocity
        acc_body = self.robot.acceleration
        
        # 使用不确定性梯度方向作为伪目标
        pos = state['position']
        uncertainty = self.uncertainty_map.get_uncertainty(pos)
        
        # 寻找最高不确定性方向
        best_dir = self._find_high_uncertainty_direction(pos, state['heading'])
        
        return {
            'lidar': lidar_normalized,
            'lidar_raw': lidar_scan,
            'velocity': vel_body,
            'acceleration': acc_body,
            'position': state['position'],
            'heading': state['heading'],
            'pseudo_goal_dir': best_dir,
            'uncertainty': uncertainty
        }
    
    def _find_high_uncertainty_direction(self, position: np.ndarray, heading: float) -> np.ndarray:
        """找到高不确定性方向"""
        # 在多个方向上采样不确定性
        num_samples = 8
        angles = np.linspace(-np.pi, np.pi, num_samples, endpoint=False)
        sample_dist = self.sensor_radius * 2
        
        best_uncertainty = -1
        best_dir = np.array([np.cos(heading), np.sin(heading)])
        
        for angle in angles:
            sample_pos = position + sample_dist * np.array([np.cos(angle), np.sin(angle)])
            
            # 检查是否在地图范围内
            if (0 < sample_pos[0] < cfg['env']['map_size'][0] and
                0 < sample_pos[1] < cfg['env']['map_size'][1]):
                u = self.uncertainty_map.get_uncertainty(sample_pos)
                if u > best_uncertainty:
                    best_uncertainty = u
                    best_dir = np.array([np.cos(angle), np.sin(angle)])
        
        return best_dir
    
    def step(self) -> tuple:
        """执行一步仿真"""
        obs = self.get_observation()
        
        # 检查是否找到目标
        pos = obs['position']
        dist_to_target = np.linalg.norm(pos - self.target_position)
        
        if dist_to_target <= self.sensor_radius:
            # 在传感器范围内，更新不确定性
            detected = dist_to_target <= self.target_radius
            self.uncertainty_map.update_observation(
                pos, detected=detected, target_position=self.target_position
            )
            
            if detected:
                self.found_target = True
                print(f"Target found at step {self.step_count}!")
                return obs, 1.0, True, {'found': True, 'steps': self.step_count}
        else:
            # 更新不确定性（未检测到）
            self.uncertainty_map.update_observation(
                pos, detected=False, target_position=self.target_position
            )
        
        # 构建网络输入
        lidar = torch.tensor(obs['lidar'], dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 使用伪目标方向
        c, s = np.cos(-obs['heading']), np.sin(-obs['heading'])
        R = np.array([[c, -s], [s, c]])
        pseudo_goal_body = R @ obs['pseudo_goal_dir'] * 10  # 放大到合适范围
        
        state = torch.tensor(
            np.concatenate([obs['velocity'], obs['acceleration'], pseudo_goal_body]),
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        # 网络推理
        with torch.no_grad():
            endstate, score = self.network.inference(lidar, state)
        
        # 选择最佳轨迹
        best_idx = score.argmin(dim=1)[0].item()
        best_endstate = endstate[0, best_idx].cpu().numpy()
        
        # 生成轨迹
        end_pos_body = best_endstate[:2]
        end_vel_body = best_endstate[2:4]
        end_acc_body = best_endstate[4:6]
        
        # 转换到世界坐标
        c, s = np.cos(obs['heading']), np.sin(obs['heading'])
        R_inv = np.array([[c, -s], [s, c]])
        
        end_pos_world = pos + R_inv @ end_pos_body
        end_vel_world = R_inv @ end_vel_body
        end_acc_world = R_inv @ end_acc_body
        
        # 使用多项式轨迹
        segment_time = cfg['trajectory']['segment_time']
        poly_solver = Poly5Solver2D(
            pos, obs['velocity'], obs['acceleration'],
            end_pos_world, end_vel_world, end_acc_world,
            segment_time
        )
        
        # 执行轨迹跟踪
        t = 0
        while t < segment_time:
            cmd_pos = poly_solver.get_position(t)
            cmd_vel = poly_solver.get_velocity(t)
            cmd_acc = poly_solver.get_acceleration(t)
            
            self.robot.step_with_control(cmd_pos, cmd_vel, cmd_acc, self.dt)
            t += self.dt
            self.step_count += 1
            
            # 检查碰撞
            if self.map_2d.get_distance(self.robot.position) < self.robot.radius:
                return obs, -1.0, True, {'collision': True, 'steps': self.step_count}
            
            # 检查步数限制
            if self.step_count >= self.max_steps:
                return obs, 0.0, True, {'timeout': True, 'steps': self.step_count}
        
        # 更新轨迹
        self.trajectory.append(self.robot.position.copy())
        
        # 更新不确定性地图
        self.uncertainty_map.compute_uncertainty()
        
        return obs, 0.0, False, {'steps': self.step_count}
    
    def run_episode(self, max_steps: int = None, visualize: bool = True) -> dict:
        """运行完整episode"""
        max_steps = max_steps or self.max_steps
        
        if visualize:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            plt.ion()
        
        done = False
        total_reward = 0.0
        
        while not done and self.step_count < max_steps:
            obs, reward, done, info = self.step()
            total_reward += reward
            
            if visualize and self.step_count % 10 == 0:
                self._visualize(axes, obs)
                plt.pause(0.01)
        
        if visualize:
            plt.ioff()
            self._visualize(axes, obs)
            plt.savefig('search_result.png')
            plt.show()
        
        return {
            'found': self.found_target,
            'steps': self.step_count,
            'total_reward': total_reward,
            'trajectory_length': len(self.trajectory)
        }
    
    def _visualize(self, axes, obs):
        """可视化"""
        for ax in axes:
            ax.clear()
        
        pos = obs['position']
        heading = obs['heading']
        
        # 障碍物地图
        ax = axes[0]
        ax.imshow(self.map_2d.grid.T, origin='lower', cmap='binary',
                 extent=[0, cfg['env']['map_size'][0], 0, cfg['env']['map_size'][1]])
        
        # 轨迹
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1.5, alpha=0.7)
        
        # 机器人
        ax.add_patch(Circle(pos, self.robot.radius, color='green', alpha=0.7))
        ax.arrow(pos[0], pos[1], np.cos(heading) * 2, np.sin(heading) * 2,
                head_width=0.5, head_length=0.3, fc='green', ec='green')
        
        # 目标
        ax.add_patch(Circle(self.target_position, self.target_radius, color='red', alpha=0.7))
        ax.plot(self.target_position[0], self.target_position[1], 'r*', markersize=15)
        
        # 传感器范围
        ax.add_patch(Circle(pos, self.sensor_radius, fill=False, color='blue', linestyle='--', alpha=0.5))
        
        ax.set_title(f'Map (Step {self.step_count})')
        ax.set_xlim(0, cfg['env']['map_size'][0])
        ax.set_ylim(0, cfg['env']['map_size'][1])
        ax.set_aspect('equal')
        
        # 不确定性地图
        ax = axes[1]
        self.uncertainty_map.compute_uncertainty()
        ax.imshow(self.uncertainty_map.uncertainty_map.T, origin='lower', cmap='hot',
                 extent=[0, cfg['env']['map_size'][0], 0, cfg['env']['map_size'][1]],
                 vmin=0, vmax=1)
        ax.plot(pos[0], pos[1], 'go', markersize=10)
        ax.plot(self.target_position[0], self.target_position[1], 'c*', markersize=15)
        ax.set_title('Uncertainty Map')
        ax.set_aspect('equal')
        
        # 激光数据
        ax = axes[2]
        angles = np.linspace(-np.pi, np.pi, len(obs['lidar_raw']), endpoint=False)
        ax.plot(np.degrees(angles), obs['lidar_raw'])
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('Range (m)')
        ax.set_title('LiDAR Scan')
        ax.set_xlim(-180, 180)
        ax.set_ylim(0, cfg['sensor']['max_range'])
        ax.grid(True)
        
        plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Search Task Test")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--map_type", type=str, default="forest", help="Map type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_vis", action="store_true", help="Disable visualization")
    args = parser.parse_args()
    
    # 查找最新模型
    if args.model is None:
        saved_dir = os.path.join(os.path.dirname(__file__), 'saved')
        if os.path.exists(saved_dir):
            search_runs = [d for d in os.listdir(saved_dir) if d.startswith('Search_')]
            if search_runs:
                latest_run = os.path.join(saved_dir, sorted(search_runs)[-1])
                models = [f for f in os.listdir(latest_run) if f.endswith('.pth')]
                if models:
                    args.model = os.path.join(latest_run, sorted(models)[-1])
                    print(f"Using latest model: {args.model}")
    
    # 创建仿真器
    sim = SearchSimulator(model_path=args.model)
    
    # 重置
    sim.reset(map_type=args.map_type, seed=args.seed)
    
    # 运行
    result = sim.run_episode(visualize=not args.no_vis)
    
    print("\n=== Search Result ===")
    print(f"Found target: {result['found']}")
    print(f"Steps: {result['steps']}")
    print(f"Trajectory length: {result['trajectory_length']}")


if __name__ == "__main__":
    main()
