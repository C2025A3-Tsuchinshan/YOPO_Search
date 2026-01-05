"""
2D YOPO 测试和可视化
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
from matplotlib.collections import LineCollection
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from simulator.map_generator import Map2D
from simulator.sensor import Lidar2D
from simulator.dynamics import Robot2D, Poly5Solver2D
from policy.network import YopoNetwork2D
from policy.primitive import LatticePrimitive2D


class YopoSimulator2D:
    """2D YOPO仿真器"""
    
    def __init__(
        self,
        model_path: str = None,
        use_gpu: bool = True,
        ideal_tracking: bool = False,
        plan_from_reference: bool = False,
        show_goal_angle: bool = False,
        goal_angle_interval: int = 1
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
        self.robot = Robot2D()
        self.lidar = Lidar2D()
        self.primitives = LatticePrimitive2D.get_instance()
        self.ideal_tracking = ideal_tracking
        self.plan_from_reference = plan_from_reference
        self.show_goal_angle = show_goal_angle
        self.goal_angle_interval = max(1, int(goal_angle_interval))
        
        # 参数
        sim_cfg = cfg['simulation']
        self.dt = sim_cfg['dt']
        self.max_steps = sim_cfg['max_steps']
        self.goal_threshold = sim_cfg['goal_threshold']
        # 碰撞恢复相关参数
        self.max_collision_attempts = sim_cfg.get('max_collision_attempts', 3)  # 最大恢复尝试次数
        
        # 状态
        self.goal = None
        self.trajectory = []
        self.current_traj_poly = None
        self.traj_start_time = 0.0
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.step_idx = 0
    
    def reset(
        self,
        map_type: str = None,
        start_pos: np.ndarray = None,
        goal_pos: np.ndarray = None,
        seed: int = None
    ):
        """重置仿真"""
        # 生成地图
        seed = seed or np.random.randint(0, 10000)
        self.map_2d = Map2D(seed=seed)
        self.map_2d.generate(map_type)
        # if self.use_true_cost:
        #     self.loss_fn.set_esdf_maps([self.map_2d.esdf], self.map_2d)
        
        # 设置起点
        if start_pos is None:
            start_pos = self.map_2d.sample_free_position(self.robot.radius)
        
        # 设置终点
        if goal_pos is None:
            for _ in range(100):
                goal_pos = self.map_2d.sample_free_position(self.robot.radius)
                if goal_pos is not None and np.linalg.norm(goal_pos - start_pos) > 10:
                    break
        
        self.goal = goal_pos
        
        # 重置机器人
        heading = np.arctan2(goal_pos[1] - start_pos[1], goal_pos[0] - start_pos[0])
        self.robot.reset(
            position=start_pos,
            velocity=np.array([0.5, 0.0]),  # 初始速度
            heading=heading
        )
        self.desire_pos = self.robot.position.copy()
        self.desire_vel = self.robot.velocity.copy()
        self.desire_acc = self.robot.acceleration.copy()
        
        # 清空轨迹
        self.trajectory = [start_pos.copy()]
        self.current_traj_poly = None
        # 重置碰撞相关状态
        self.collision_count = 0
        self.last_collision_pos = None

        return self.get_observation()
    
    def get_observation(self) -> dict:
        """获取观测"""
        state = self.robot.get_state()
        
        # 激光扫描
        lidar_scan = self.lidar.scan(
            self.map_2d, state['position'], state['heading'], add_noise=False
        )
        lidar_normalized = self.lidar.normalize_ranges(lidar_scan)
        
        # 机体状态
        state_body = self.robot.get_body_state(self.goal)
        
        return {
            'lidar': lidar_normalized,
            'lidar_raw': lidar_scan,
            'state_body': state_body,
            'position': state['position'],
            'velocity': state['velocity'],
            'acceleration': state['acceleration'],
            'heading': state['heading'],
            'goal': self.goal
        }

    def _compute_goal_angle(self, obs: dict) -> dict:
        goal_body = obs['state_body'][4:6]
        angle_rad = float(np.arctan2(goal_body[1], goal_body[0]))
        angle_deg = np.degrees(angle_rad)
        goal_dir = obs['goal'] - obs['position']
        heading_vec = np.array([np.cos(obs['heading']), np.sin(obs['heading'])])
        dot = float(np.dot(goal_dir, heading_vec))
        return {
            'goal_body': goal_body,
            'angle_deg': angle_deg,
            'dot': dot,
            'dist': float(np.linalg.norm(goal_dir))
        }

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _blend_heading(self, vel_cmd: np.ndarray, goal_dir: np.ndarray, last_yaw: float) -> float:
        vel_norm = np.linalg.norm(vel_cmd)
        goal_norm = np.linalg.norm(goal_dir)

        if vel_norm < 1e-6 and goal_norm < 1e-6:
            return last_yaw

        vel_unit = vel_cmd / vel_norm if vel_norm > 1e-6 else np.zeros(2)
        goal_unit = goal_dir / goal_norm if goal_norm > 1e-6 else np.zeros(2)

        alpha = np.clip(vel_norm / max(self.robot.max_vel, 1e-6), 0.0, 1.0)
        blend = alpha * vel_unit + (1.0 - alpha) * goal_unit
        if np.linalg.norm(blend) < 1e-6:
            blend = goal_unit if goal_norm > 1e-6 else vel_unit

        target_yaw = float(np.arctan2(blend[1], blend[0]))
        yaw_diff = self._wrap_angle(target_yaw - last_yaw)
        max_change = self.robot.max_omega * self.dt
        yaw = self._wrap_angle(last_yaw + np.clip(yaw_diff, -max_change, max_change))
        return yaw

    def _compute_body_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        heading: float,
        goal: np.ndarray
    ) -> np.ndarray:
        c, s = np.cos(-heading), np.sin(-heading)
        R = np.array([[c, -s], [s, c]])
        vel_body = R @ velocity
        acc_body = R @ acceleration
        goal_body = R @ (goal - position)
        return np.concatenate([vel_body, acc_body, goal_body])
    
    @torch.no_grad()
    def plan(self, obs: dict) -> dict:
        """
        规划轨迹，并对所有候选进行碰撞检测过滤
        
        关键改进:
        1. 添加轨迹长度奖励，避免过短轨迹
        2. 当前方被堵时，优先选择侧向绕行轨迹
        3. 结合网络 score + 到目标距离 + 轨迹长度 的综合评分
        """
        # 准备输入
        lidar = torch.tensor(obs['lidar'], dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.plan_from_reference and self.desire_pos is not None:
            start_pos = self.desire_pos
            start_vel = self.desire_vel
            start_acc = self.desire_acc
            state_body = self._compute_body_state(start_pos, start_vel, start_acc, obs['heading'], obs['goal'])
        else:
            start_pos = obs['position']
            start_vel = obs['velocity']
            start_acc = obs['acceleration']
            state_body = obs['state_body']
        state = torch.tensor(state_body, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # network inference
        endstate, score = self.network.inference(lidar, state)
        # endstate: [1, N, 6], score: [1, N]
        endstate_np = endstate[0].cpu().numpy()
        scores_np = score[0].cpu().numpy()
        num_candidates = endstate_np.shape[0]

        # transform to world and collision check
        heading = obs['heading']
        position = start_pos
        acc0 = start_acc
        goal = obs['goal']
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s], [s, c]])

        safe_flags = np.ones(num_candidates, dtype=bool)
        end_positions_world = []  # 保存所有终点位置
        traj_lengths = []  # 轨迹长度
        
        # 对所有候选轨迹进行碰撞检测
        collision_check_points = 20  # 碰撞检测采样点数
        for i, endstate in enumerate(endstate_np):
            end_pos_body = endstate[:2]
            end_vel_body = endstate[2:4]
            end_acc_body = endstate[4:6]
            end_pos_world = R @ end_pos_body + position
            end_vel_world = R @ end_vel_body
            end_acc_world = R @ end_acc_body
            end_positions_world.append(end_pos_world)
            traj_lengths.append(np.linalg.norm(end_pos_body))  # 机体系下的轨迹长度
            
            # 构建轨迹多项式
            traj_check = Poly5Solver2D(
                pos0=position,
                vel0=start_vel,
                acc0=acc0,
                pos1=end_pos_world,
                vel1=end_vel_world,
                acc1=end_acc_world,
                T=self.primitives.segment_time
            )
            
            # 采样检测碰撞
            for t in np.linspace(0, self.primitives.segment_time, collision_check_points):
                pos_check = traj_check.get_position(t)
                if not self.map_2d.is_valid_position(pos_check, self.robot.radius):
                    safe_flags[i] = False
                    break
        
        end_positions_world = np.array(end_positions_world)
        traj_lengths = np.array(traj_lengths)
        
        # 计算到目标的距离
        dist_to_goal = np.linalg.norm(position - goal)
        goal_dir = goal - position
        goal_dir_norm = goal_dir / (np.linalg.norm(goal_dir) + 1e-8)
        
        # 检查是否所有轨迹都很短（可能卡住了）
        all_short = np.all(traj_lengths < 1.0)  # 如果所有轨迹长度都小于 1m
        
        # === 综合评分系统 ===
        # 对每个安全轨迹计算综合得分，结合:
        # 1. 网络 score (安全+平滑)
        # 2. 轨迹终点到目标的距离
        # 3. 轨迹长度奖励 (避免过短轨迹)
        # 4. 绕行奖励 (当前方被堵时)
        
        def compute_combined_score(indices):
            """计算安全轨迹的综合得分"""
            if len(indices) == 0:
                return np.array([])
            
            # 基础分: 网络 score (归一化)
            net_scores = scores_np[indices]
            net_scores_norm = (net_scores - net_scores.min()) / (net_scores.max() - net_scores.min() + 1e-8)
            
            # 目标接近度: 轨迹终点到目标的距离 (归一化)
            end_to_goal_dist = np.linalg.norm(end_positions_world[indices] - goal, axis=1)
            goal_score = end_to_goal_dist / (self.primitives.radio_range * 2 + 1e-8)
            
            # 轨迹长度奖励: 长轨迹优先 (避免卡住时选择过短轨迹)
            lengths = traj_lengths[indices]
            max_length = self.primitives.radio_range * 2
            length_reward = 1.0 - (lengths / max_length)  # 越长奖励越大 (负向得分)
            
            # 沿目标方向的前进距离
            end_pos_body_list = endstate_np[indices, :2]
            progress = np.sum(end_pos_body_list * goal_dir_norm, axis=1)  # 投影到目标方向
            progress_score = -progress / max_length  # 前进越多得分越低
            
            # 检测是否前方被堵：正面轨迹 (中间的) 是否都不安全
            num_candidates = len(safe_flags)
            center_idx = num_candidates // 2
            front_blocked = not safe_flags[center_idx]  # 正前方轨迹不安全
            
            # 动态权重调整
            if front_blocked and dist_to_goal > self.primitives.radio_range:
                # 前方被堵时，更重视轨迹长度和侧向绕行
                w_net = 0.2
                w_goal = 0.3
                w_length = 0.3
                w_progress = 0.2
            elif dist_to_goal < self.primitives.radio_range:
                # 接近目标时，更重视目标距离
                w_net = 0.1
                w_goal = 0.7
                w_length = 0.1
                w_progress = 0.1
            else:
                # 正常情况
                w_net = 0.4
                w_goal = 0.3
                w_length = 0.15
                w_progress = 0.15
            
            combined = w_net * net_scores_norm + w_goal * goal_score + w_length * length_reward + w_progress * progress_score
            return combined
        
        # === "直接到目标"策略 ===
        # 当距离目标非常近（< 3.0m）或者卡住时，直接规划到目标点
        direct_to_goal = False
        direct_goal_traj = None
        if dist_to_goal < 3.0 or (all_short and dist_to_goal < self.primitives.radio_range):
            # 构建直接到目标的轨迹
            direct_traj = Poly5Solver2D(
                pos0=position,
                vel0=start_vel,
                acc0=acc0,
                pos1=goal,
                vel1=np.zeros(2),
                acc1=np.zeros(2),
                T=self.primitives.segment_time
            )
            
            # 检查这个轨迹是否安全
            is_safe = True
            for t in np.linspace(0, self.primitives.segment_time, 20):
                pos_check = direct_traj.get_position(t)
                if not self.map_2d.is_valid_position(pos_check, self.robot.radius):
                    is_safe = False
                    break
            
            if is_safe:
                direct_to_goal = True
                direct_goal_traj = direct_traj
        
        # 选择最优安全轨迹
        safe_indices = np.where(safe_flags)[0]
        if direct_to_goal:
            # 直接使用到目标的轨迹
            chosen_idx = -1  # 特殊标记表示直接到目标
            no_safe = False
            traj_poly = direct_goal_traj
            end_pos_world = goal
            end_vel_world = np.zeros(2)
        elif len(safe_indices) > 0:
            # 使用综合评分选择最优轨迹
            combined_scores = compute_combined_score(safe_indices)
            best_safe_local_idx = np.argmin(combined_scores)
            best_safe_idx = safe_indices[best_safe_local_idx]
            chosen_idx = int(best_safe_idx)
            no_safe = False
            
            # 构建最终选择的轨迹
            chosen_end = endstate_np[chosen_idx]
            end_pos_body = chosen_end[:2]
            end_vel_body = chosen_end[2:4]
            end_acc_body = chosen_end[4:6]
            end_pos_world = R @ end_pos_body + position
            end_vel_world = R @ end_vel_body
            end_acc_world = R @ end_acc_body

            traj_poly = Poly5Solver2D(
                pos0=position,
                vel0=start_vel,
                acc0=acc0,
                pos1=end_pos_world,
                vel1=end_vel_world,
                acc1=end_acc_world,
                T=self.primitives.segment_time
            )
        else:
            # 没有安全轨迹，选择得分最低的（可能是碰撞轨迹）
            chosen_idx = int(np.argmin(scores_np))
            no_safe = True
            
            # 构建最终选择的轨迹
            chosen_end = endstate_np[chosen_idx]
            end_pos_body = chosen_end[:2]
            end_vel_body = chosen_end[2:4]
            end_acc_body = chosen_end[4:6]
            end_pos_world = R @ end_pos_body + position
            end_vel_world = R @ end_vel_body
            end_acc_world = R @ end_acc_body

            traj_poly = Poly5Solver2D(
                pos0=position,
                vel0=start_vel,
                acc0=acc0,
                pos1=end_pos_world,
                vel1=end_vel_world,
                acc1=end_acc_world,
                T=self.primitives.segment_time
            )

        return {
            'trajectory': traj_poly,
            'end_pos': end_pos_world,
            'end_vel': end_vel_world,
            'score': scores_np,
            'best_idx': chosen_idx,
            'all_endstates': endstate_np,
            'safe_flags': safe_flags,
            'no_safe': no_safe,
            'direct_to_goal': direct_to_goal
        }
    
    def step(self, plan_result: dict = None) -> tuple:
        """执行一步仿真"""
        # 如果没有轨迹或需要重新规划
        if plan_result is not None:
            self.current_traj_poly = plan_result['trajectory']
            self.traj_start_time = 0.0
        
        if self.current_traj_poly is None:
            obs = self.get_observation()
            plan_result = self.plan(obs)
            self.current_traj_poly = plan_result['trajectory']
            self.traj_start_time = 0.0
        
        # 从轨迹获取控制命令
        t = self.traj_start_time + self.dt
        if t > self.primitives.segment_time:
            # 轨迹结束，重新规划
            obs = self.get_observation()
            plan_result = self.plan(obs)
            self.current_traj_poly = plan_result['trajectory']
            self.traj_start_time = 0.0
            t = self.dt
        
        pos_cmd = self.current_traj_poly.get_position(t)
        vel_cmd = self.current_traj_poly.get_velocity(t)
        acc_cmd = self.current_traj_poly.get_acceleration(t)
        
        # 执行控制
        if self.ideal_tracking:
            self.robot.position = pos_cmd.copy()
            self.robot.velocity = vel_cmd.copy()
            self.robot.acceleration = acc_cmd.copy()
            goal_dir = self.goal - self.robot.position
            self.robot.heading = self._blend_heading(vel_cmd, goal_dir, self.robot.heading)
        else:
            self.robot.step_with_control(pos_cmd, vel_cmd, acc_cmd, self.dt)

        self.desire_pos = pos_cmd.copy()
        self.desire_vel = vel_cmd.copy()
        self.desire_acc = acc_cmd.copy()
        
        self.traj_start_time = t
        self.trajectory.append(self.robot.position.copy())
        
        # 检查终止条件
        dist_to_goal = np.linalg.norm(self.robot.position - self.goal)
        reached_goal = dist_to_goal < self.goal_threshold
        collision = not self.map_2d.is_valid_position(self.robot.position, self.robot.radius)
        
        # 构建 info
        info = {
            'dist_to_goal': dist_to_goal,
            'collision': collision,
            'success': reached_goal and not collision
        }
        
        done = reached_goal and not collision

        if collision:
            # 增加计数并记录碰撞位置
            self.collision_count = getattr(self, 'collision_count', 0) + 1
            self.last_collision_pos = self.robot.position.copy()

            # 回退到上一个安全历史点（如果有）并裁剪轨迹
            last_safe_pos = None
            for idx in range(len(self.trajectory) - 1, -1, -1):
                if self.map_2d.is_valid_position(self.trajectory[idx], self.robot.radius):
                    last_safe_pos = self.trajectory[idx]
                    self.trajectory = self.trajectory[:idx+1]
                    break
            if last_safe_pos is None:
                last_safe_pos = self.trajectory[0]

            # 回退机器人并准备重新规划
            self.robot.position = last_safe_pos.copy()
            self.robot.velocity = np.zeros_like(self.robot.velocity)
            self.current_traj_poly = None
            self.traj_start_time = 0.0

            # 判断是否超过允许恢复次数
            if self.collision_count >= self.max_collision_attempts:
                done = True
                info['failure_reason'] = 'collision_limit'
                info['success'] = False
            else:
                info['recovery'] = True
                info['success'] = False
        else:
            # 正常步进，清除恢复状态
            self.collision_count = 0
            self.last_collision_pos = None

        return self.get_observation(), done, info
    
    def run_episode(self, max_steps: int = None, visualize: bool = True) -> dict:
        """运行一个episode"""
        max_steps = max_steps or self.max_steps
        self.step_idx = 0
        
        obs = self.get_observation()
        
        if visualize:
            fig, axes = self.setup_visualization()
            plt.show(block=False)  # 非阻塞显示窗口
        
        for step in range(max_steps):
            self.step_idx = step
            # 规划
            plan_result = self.plan(obs)
            
            # 可视化（局部+全局）
            if visualize:
                self.update_visualization(axes, obs, plan_result)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)
            if self.show_goal_angle and (step % self.goal_angle_interval == 0):
                info_angle = self._compute_goal_angle(obs)
                print(
                    f"[step {step}] goal_body_angle={info_angle['angle_deg']:.1f}deg "
                    f"dot={info_angle['dot']:.2f} dist={info_angle['dist']:.2f}m"
                )
            
            # 执行
            obs, done, info = self.step(plan_result)

            # 如果触发了碰撞恢复，打印信息（便于调试）
            if info.get('recovery', False):
                print(f"Collision detected. Recovery attempt {self.collision_count}/{self.max_collision_attempts}. Rolling back to safe point.")

            if done:
                break
        
        if visualize:
            plt.ioff()
            plt.close(fig)  # 关闭窗口以避免阻塞
        
        return {
            'trajectory': np.array(self.trajectory),
            'success': info.get('success', False),
            'steps': step + 1,
            'final_dist': info['dist_to_goal']
        }
    
    def setup_visualization(self):
        """设置可视化（左：局部视图，右：全局视图）"""
        fig, (ax_local, ax_global) = plt.subplots(1, 2, figsize=(16, 8))
        ax_local.set_aspect('equal')
        ax_global.set_aspect('equal')
        ax_global.set_title('Global view')
        ax_global.set_xlim(0, self.map_2d.size[0])
        ax_global.set_ylim(0, self.map_2d.size[1])
        return fig, (ax_local, ax_global)
    
    def update_visualization(self, axes, obs: dict, plan_result: dict):
        """更新可视化（axes: (ax_local, ax_global))"""
        ax, ax_global = axes
        ax.clear()
        ax_global.clear()
        
        # 绘制地图
        ax.imshow(
            self.map_2d.grid.T, origin='lower', cmap='binary',
            extent=[0, self.map_2d.size[0], 0, self.map_2d.size[1]],
            alpha=0.5
        )
        ax_global.imshow(
            self.map_2d.grid.T, origin='lower', cmap='binary',
            extent=[0, self.map_2d.size[0], 0, self.map_2d.size[1]],
            alpha=0.3
        )
        
        # 绘制目标
        ax.plot(self.goal[0], self.goal[1], 'g*', markersize=20, label='Goal')
        ax_global.plot(self.goal[0], self.goal[1], 'g*', markersize=14, label='Goal')
        
        # 绘制机器人
        pos = obs['position']
        heading = obs['heading']
        
        robot_circle = Circle(pos, self.robot.radius, fill=False, color='blue', linewidth=2)
        ax.add_patch(robot_circle)
        ax.arrow(pos[0], pos[1], 
            np.cos(heading) * self.robot.radius * 1.5,
            np.sin(heading) * self.robot.radius * 1.5,
            head_width=0.2, head_length=0.1, fc='blue', ec='blue')
        # 全局位置
        robot_circle_global = Circle(pos, self.robot.radius, fill=False, color='blue', linewidth=1.5)
        ax_global.add_patch(robot_circle_global)
        ax_global.arrow(pos[0], pos[1],
            np.cos(heading) * self.robot.radius * 1.2,
            np.sin(heading) * self.robot.radius * 1.2,
            head_width=0.15, head_length=0.08, fc='blue', ec='blue')
        
        # 绘制激光点云
        points = self.lidar.scan_to_points(obs['lidar_raw'], pos, heading)
        ax.scatter(points[:, 0], points[:, 1], c='red', s=2, alpha=0.5)
        
        # 绘制所有候选轨迹
        all_endstates = plan_result['all_endstates']  # [N, 6]
        scores = plan_result['score']
        best_idx = plan_result['best_idx']

        for i, (endstate, score) in enumerate(zip(all_endstates, scores)):
            end_pos_body = endstate[:2]
            c, s = np.cos(heading), np.sin(heading)
            R = np.array([[c, -s], [s, c]])
            end_pos_world = R @ end_pos_body + pos
            end_vel_body = endstate[2:4]
            end_acc_body = endstate[4:6]
            end_vel_world = R @ end_vel_body
            end_acc_world = R @ end_acc_body

            # 使用 Poly5Solver2D 绘制完整轨迹段
            traj_poly = Poly5Solver2D(
                pos0=pos,
                vel0=obs['velocity'],
                acc0=np.zeros(2),
                pos1=end_pos_world,
                vel1=np.zeros(2),  # 假设终点速度为零
                acc1=np.zeros(2),  # 假设终点加速度为零
                T=self.primitives.segment_time
            )
            if i == best_idx and plan_result.get('trajectory') is not None:
                traj_poly = plan_result['trajectory']
            else:
                traj_poly = Poly5Solver2D(
                    pos0=pos,
                    vel0=obs['velocity'],
                    acc0=np.zeros(2),
                    pos1=end_pos_world,
                    vel1=end_vel_world,
                    acc1=end_acc_world,
                    T=self.primitives.segment_time
                )

            # 生成轨迹点
            t_vals = np.linspace(0, self.primitives.segment_time, num=50)
            traj_points = np.array([traj_poly.get_position(t) for t in t_vals])

            # 根据安全标志改变颜色/样式
            unsafe = False
            if 'safe_flags' in plan_result:
                unsafe = not bool(plan_result['safe_flags'][i])

            if i == best_idx:
                color = 'green'
                lw = 2
                alpha = 1.0
                ls = '-'
            elif unsafe:
                color = 'red'
                lw = 1
                alpha = 0.4
                ls = '--'
            else:
                color = 'gray'
                lw = 1
                alpha = 0.3
                ls = '-'

            ax.plot(traj_points[:, 0], traj_points[:, 1], linestyle=ls, color=color, alpha=alpha, linewidth=lw)

        # 如果存在最近碰撞位置，绘制标记（便于调试）
        if getattr(self, 'last_collision_pos', None) is not None:
            cp = self.last_collision_pos
            ax.plot(cp[0], cp[1], 'rx', markersize=12, markeredgewidth=2, label='Collision')
            ax_global.plot(cp[0], cp[1], 'rx', markersize=10, markeredgewidth=1)

        # 绘制历史轨迹
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=0.5)
            ax_global.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1.5, alpha=0.8, label='Trajectory' )
        
        # 设置范围
        margin = 5
        ax.set_xlim(pos[0] - 15, pos[0] + 15)
        ax.set_ylim(pos[1] - 15, pos[1] + 15)
        ax_global.set_xlim(0, self.map_2d.size[0])
        ax_global.set_ylim(0, self.map_2d.size[1])
        ax_global.legend(loc='upper right')
        
        ax.set_title(f'YOPO 2D | Dist to goal: {np.linalg.norm(pos - self.goal):.2f}m')
        if self.show_goal_angle:
            info_angle = self._compute_goal_angle(obs)
            ax.text(
                0.02, 0.98,
                f"goal_body_angle: {info_angle['angle_deg']:.1f}deg\n"
                f"dot: {info_angle['dot']:.2f}",
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
        ax.legend(loc='upper right')


def test():
    """测试入口"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Model path')
    parser.add_argument('--map_type', type=str, default='forest', help='Map type')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--no_vis', action='store_true', help='Disable visualization')
    parser.add_argument('--ideal_tracking', action='store_true', help='Use ideal trajectory tracking')
    parser.add_argument('--plan_from_reference', action='store_true', help='Plan from desired state')
    parser.add_argument('--show_goal_angle', action='store_true', help='Print/visualize goal-body angle')
    parser.add_argument('--goal_angle_interval', type=int, default=1, help='Print interval for goal-body angle')
    args = parser.parse_args()
    
    # 查找最新模型
    if args.model is None:
        saved_dir = os.path.join(os.path.dirname(__file__), 'saved')
        if os.path.exists(saved_dir):
            runs = sorted([d for d in os.listdir(saved_dir) if d.startswith('YOPO2D_')])
            if runs:
                latest_run = os.path.join(saved_dir, runs[-1])
                models = [f for f in os.listdir(latest_run) if f.endswith('.pth')]
                if models:
                    args.model = os.path.join(latest_run, sorted(models)[-1])
    
    sim = YopoSimulator2D(
        model_path=args.model,
        ideal_tracking=args.ideal_tracking,
        plan_from_reference=args.plan_from_reference,
        show_goal_angle=args.show_goal_angle,
        goal_angle_interval=args.goal_angle_interval
    )
    sim.reset(map_type=args.map_type, seed=args.seed)
    
    result = sim.run_episode(visualize=not args.no_vis)
    
    print(f"\n=== Results ===")
    print(f"Success: {result['success']}")
    print(f"Steps: {result['steps']}")
    print(f"Final distance to goal: {result['final_dist']:.2f}m")


if __name__ == "__main__":
    plt.ion()  # 启用交互模式
    test()
