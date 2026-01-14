"""
2D机器人动力学模型
简化的二阶积分器模型
"""

import numpy as np
from typing import Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg


class Robot2D:
    """2D机器人 (二阶积分器模型)"""
    
    def __init__(
        self,
        radius: float = None,
        max_vel: float = None,
        max_acc: float = None,
        max_omega: float = None
    ):
        """
        Args:
            radius: 机器人半径
            max_vel: 最大速度
            max_acc: 最大加速度
            max_omega: 最大角速度
        """
        self.radius = radius or cfg['robot']['radius']
        self.max_vel = max_vel or cfg['robot']['max_vel']
        self.max_acc = max_acc or cfg['robot']['max_acc']
        self.max_omega = max_omega or cfg['robot']['max_omega']
        
        # 状态: [x, y, vx, vy, ax, ay, heading]
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.heading = 0.0  # radians
    
    def reset(
        self,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        acceleration: np.ndarray = None,
        heading: float = None
    ):
        """重置状态"""
        self.position = np.asarray(position) if position is not None else np.zeros(2)
        self.velocity = np.asarray(velocity) if velocity is not None else np.zeros(2)
        self.acceleration = np.asarray(acceleration) if acceleration is not None else np.zeros(2)
        self.heading = heading if heading is not None else 0.0
    
    def get_state(self) -> dict:
        """获取当前状态"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'acceleration': self.acceleration.copy(),
            'heading': self.heading
        }
    
    def set_state(
        self,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        acceleration: np.ndarray = None,
        heading: float = None
    ):
        """设置状态"""
        if position is not None:
            self.position = np.asarray(position)
        if velocity is not None:
            self.velocity = np.asarray(velocity)
        if acceleration is not None:
            self.acceleration = np.asarray(acceleration)
        if heading is not None:
            self.heading = heading
    
    def step(self, desired_acc: np.ndarray, dt: float) -> dict:
        """
        执行一步仿真
        
        Args:
            desired_acc: 期望加速度 [ax, ay]
            dt: 时间步长
            
        Returns:
            state: 新状态
        """
        # 限制加速度
        acc_norm = np.linalg.norm(desired_acc)
        if acc_norm > self.max_acc:
            desired_acc = desired_acc * self.max_acc / acc_norm
        
        # 更新加速度
        self.acceleration = desired_acc
        
        # 更新速度
        new_velocity = self.velocity + self.acceleration * dt
        vel_norm = np.linalg.norm(new_velocity)
        if vel_norm > self.max_vel:
            new_velocity = new_velocity * self.max_vel / vel_norm
        
        # 更新位置 (二阶积分)
        self.position = self.position + self.velocity * dt + 0.5 * self.acceleration * dt * dt
        self.velocity = new_velocity
        
        # 更新朝向 (根据速度方向)
        if vel_norm > 0.1:
            target_heading = np.arctan2(self.velocity[1], self.velocity[0])
            heading_diff = self._wrap_angle(target_heading - self.heading)
            max_heading_change = self.max_omega * dt
            heading_change = np.clip(heading_diff, -max_heading_change, max_heading_change)
            self.heading = self._wrap_angle(self.heading + heading_change)
        
        return self.get_state()
    
    def step_with_control(
        self,
        position_cmd: np.ndarray,
        velocity_cmd: np.ndarray,
        acceleration_cmd: np.ndarray,
        dt: float,
        kp: float = 10.0,
        kv: float = 3.0
    ) -> dict:
        """
        使用位置-速度控制执行一步仿真
        
        Args:
            position_cmd: 期望位置
            velocity_cmd: 期望速度
            acceleration_cmd: 前馈加速度
            dt: 时间步长
            kp, kv: 控制增益
            
        Returns:
            state: 新状态
        """
        # PD控制计算期望加速度
        pos_error = position_cmd - self.position
        vel_error = velocity_cmd - self.velocity
        
        desired_acc = kp * pos_error + kv * vel_error + acceleration_cmd
        
        return self.step(desired_acc, dt)
    
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """将角度限制在 [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def get_body_state(self, goal: np.ndarray) -> np.ndarray:
        """
        获取机体坐标系下的状态
        
        Args:
            goal: 世界坐标系下的目标位置
            
        Returns:
            state: [vx_b, vy_b, ax_b, ay_b, goal_x_b, goal_y_b]
        """
        # 旋转矩阵 (世界到机体)
        c, s = np.cos(-self.heading), np.sin(-self.heading)
        R = np.array([[c, -s], [s, c]])
        
        # 速度和加速度转到机体系
        vel_body = R @ self.velocity
        acc_body = R @ self.acceleration
        
        # 目标相对位置转到机体系
        goal_rel = goal - self.position
        goal_body = R @ goal_rel
        
        return np.concatenate([vel_body, acc_body, goal_body])


class Poly4Solver2D:
    """2D四次多项式轨迹求解器"""
    
    def __init__(
        self,
        pos0: np.ndarray,
        vel0: np.ndarray,
        acc0: np.ndarray,
        pos1: np.ndarray,
        vel1: np.ndarray,
        T: float
    ):
        """
        Args:
            pos0: 起始位置 [2]
            vel0: 起始速度 [2]
            acc0: 起始加速度 [2]
            pos1: 终止位置 [2]
            vel1: 终止速度 [2]
            T: 轨迹时间
        """
        self.T = T
        self.dim = 2
        
        # 对每个维度求解系数
        self.coeffs = []
        for d in range(self.dim):
            c = self._solve_coeffs(
                pos0[d], vel0[d], acc0[d],
                pos1[d], vel1[d], T
            )
            self.coeffs.append(c)
        self.coeffs = np.array(self.coeffs)  # [2, 5]
    
    def _solve_coeffs(self, p0, v0, a0, p1, v1, T):
        """求解单维度系数"""
        # 边界条件: p(0)=p0, v(0)=v0, a(0)=a0, p(T)=p1, v(T)=v1
        # p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4
        
        c0 = p0
        c1 = v0
        c2 = a0 / 2
        
        # 求解 c3, c4
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        
        # p(T) = c0 + c1*T + c2*T^2 + c3*T^3 + c4*T^4 = p1
        # v(T) = c1 + 2*c2*T + 3*c3*T^2 + 4*c4*T^3 = v1
        
        A = np.array([
            [T3, T4],
            [3*T2, 4*T3]
        ])
        b = np.array([
            p1 - c0 - c1*T - c2*T2,
            v1 - c1 - 2*c2*T
        ])
        
        c34 = np.linalg.solve(A, b)
        
        return np.array([c0, c1, c2, c34[0], c34[1]])
    
    def get_position(self, t: float) -> np.ndarray:
        """获取位置"""
        t = np.clip(t, 0, self.T)
        tv = np.array([1, t, t**2, t**3, t**4])
        return self.coeffs @ tv
    
    def get_velocity(self, t: float) -> np.ndarray:
        """获取速度"""
        t = np.clip(t, 0, self.T)
        tv = np.array([0, 1, 2*t, 3*t**2, 4*t**3])
        return self.coeffs @ tv
    
    def get_acceleration(self, t: float) -> np.ndarray:
        """获取加速度"""
        t = np.clip(t, 0, self.T)
        tv = np.array([0, 0, 2, 6*t, 12*t**2])
        return self.coeffs @ tv
    
    def sample_trajectory(self, num_points: int = 50) -> dict:
        """采样轨迹点"""
        t = np.linspace(0, self.T, num_points)
        positions = np.array([self.get_position(ti) for ti in t])
        velocities = np.array([self.get_velocity(ti) for ti in t])
        accelerations = np.array([self.get_acceleration(ti) for ti in t])
        
        return {
            't': t,
            'position': positions,
            'velocity': velocities,
            'acceleration': accelerations
        }


class Poly5Solver2D:
    """2D五次多项式轨迹求解器"""
    
    def __init__(
        self,
        pos0: np.ndarray,
        vel0: np.ndarray,
        acc0: np.ndarray,
        pos1: np.ndarray,
        vel1: np.ndarray,
        acc1: np.ndarray,
        T: float
    ):
        """
        Args:
            pos0, vel0, acc0: 起始状态
            pos1, vel1, acc1: 终止状态
            T: 轨迹时间
        """
        self.T = T
        self.dim = 2
        
        # 系数矩阵 (来自原始3D版本的简化)
        t = T
        Coef_inv = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1/2, 0, 0, 0],
            [-10/t**3, -6/t**2, -3/(2*t), 10/t**3, -4/t**2, 1/(2*t)],
            [15/t**4, 8/t**3, 3/(2*t**2), -15/t**4, 7/t**3, -1/t**2],
            [-6/t**5, -3/t**4, -1/(2*t**3), 6/t**5, -3/t**4, 1/(2*t**3)]
        ])
        
        # 对每个维度求解系数
        self.coeffs = []
        for d in range(self.dim):
            state = np.array([pos0[d], vel0[d], acc0[d], pos1[d], vel1[d], acc1[d]])
            c = Coef_inv @ state
            self.coeffs.append(c)
        self.coeffs = np.array(self.coeffs)  # [2, 6]
    
    def get_position(self, t: float) -> np.ndarray:
        """获取位置"""
        t = np.clip(t, 0, self.T)
        tv = np.array([1, t, t**2, t**3, t**4, t**5])
        return self.coeffs @ tv
    
    def get_velocity(self, t: float) -> np.ndarray:
        """获取速度"""
        t = np.clip(t, 0, self.T)
        tv = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
        return self.coeffs @ tv
    
    def get_acceleration(self, t: float) -> np.ndarray:
        """获取加速度"""
        t = np.clip(t, 0, self.T)
        tv = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3])
        return self.coeffs @ tv
    
    def get_jerk(self, t: float) -> np.ndarray:
        """获取加加速度"""
        t = np.clip(t, 0, self.T)
        tv = np.array([0, 0, 0, 6, 24*t, 60*t**2])
        return self.coeffs @ tv
    
    def sample_trajectory(self, num_points: int = 50) -> dict:
        """采样轨迹点"""
        t = np.linspace(0, self.T, num_points)
        positions = np.array([self.get_position(ti) for ti in t])
        velocities = np.array([self.get_velocity(ti) for ti in t])
        accelerations = np.array([self.get_acceleration(ti) for ti in t])
        
        return {
            't': t,
            'position': positions,
            'velocity': velocities,
            'acceleration': accelerations
        }

    def evaluate(self, t: float) -> tuple:
        """Evaluate trajectory at time t, return position, velocity, acceleration.

        Args:
            t: time (seconds)

        Returns:
            (pos, vel, acc): each is ndarray shape (2,)
        """
        pos = self.get_position(t)
        vel = self.get_velocity(t)
        acc = self.get_acceleration(t)
        return pos, vel, acc


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 测试轨迹求解器
    pos0 = np.array([0.0, 0.0])
    vel0 = np.array([1.0, 0.0])
    acc0 = np.array([0.0, 0.0])
    pos1 = np.array([5.0, 2.0])
    vel1 = np.array([1.0, 0.5])
    acc1 = np.array([0.0, 0.0])
    T = 2.0
    
    solver = Poly5Solver2D(pos0, vel0, acc0, pos1, vel1, acc1, T)
    traj = solver.sample_trajectory()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    ax = axes[0, 0]
    ax.plot(traj['position'][:, 0], traj['position'][:, 1], 'b-')
    ax.plot(pos0[0], pos0[1], 'go', markersize=10, label='Start')
    ax.plot(pos1[0], pos1[1], 'ro', markersize=10, label='End')
    ax.set_title('Trajectory')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True)
    
    ax = axes[0, 1]
    ax.plot(traj['t'], np.linalg.norm(traj['velocity'], axis=1))
    ax.set_title('Speed')
    ax.set_xlabel('Time')
    ax.grid(True)
    
    ax = axes[1, 0]
    ax.plot(traj['t'], traj['velocity'][:, 0], label='vx')
    ax.plot(traj['t'], traj['velocity'][:, 1], label='vy')
    ax.set_title('Velocity')
    ax.legend()
    ax.grid(True)
    
    ax = axes[1, 1]
    ax.plot(traj['t'], traj['acceleration'][:, 0], label='ax')
    ax.plot(traj['t'], traj['acceleration'][:, 1], label='ay')
    ax.set_title('Acceleration')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('trajectory_test.png')
    plt.show()
