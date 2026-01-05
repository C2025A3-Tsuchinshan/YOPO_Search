import os
import yaml
from typing import Any, Dict


class Config:
    """
    全局配置管理器 (对齐 YOPO_Sim 的 config.py)
    
    自动计算派生参数:
    - goal_length = 2 * radio_range
    - sgm_time = 2 * radio_range / vel_max_train
    - traj_num = horizon_num * vertical_num
    """
    
    _instance = None
    _data: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """加载配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            self._data = yaml.safe_load(f)
        
        # 计算派生参数 (对齐 YOPO_Sim)
        self._compute_derived_params()
    
    def _compute_derived_params(self):
        """计算派生参数 (对齐 YOPO_Sim/YOPO/config/config.py)"""
        traj = self._data['trajectory']
        robot = self._data['robot']
        train = self._data['training']
        
        # 使用训练速度计算 (对齐 YOPO_Sim)
        vel_max_train = float(train.get('vel_max_train', robot['max_vel']))
        
        # radio_range: 规划半径 (对齐 YOPO_Sim)
        radio_range = float(traj.get('radio_range', traj['planning_horizon'] / 2))
        traj['radio_range'] = radio_range
        
        # goal_length = 2 * radio_range (对齐 YOPO_Sim)
        train['goal_length'] = 2.0 * radio_range
        
        # segment_time = 2 * radio_range / vel_max_train (对齐 YOPO_Sim sgm_time)
        traj['segment_time'] = 2.0 * radio_range / vel_max_train
        
        # 基元数量
        horizon_num = int(traj.get('horizon_num', traj.get('num_directions', 5)))
        vertical_num = int(traj.get('vertical_num', 1))
        traj['horizon_num'] = horizon_num
        traj['vertical_num'] = vertical_num
        traj['num_primitives'] = horizon_num * vertical_num
        
        # anchor FOV (对齐 YOPO_Sim horizon_anchor_fov)
        # 注意: direction_fov 是总覆盖角度，anchor_fov 是每个 anchor 的 FOV
        if 'anchor_fov' not in traj:
            traj['anchor_fov'] = 30.0  # 默认值对齐 YOPO_Sim
        
        # 标记训练模式 (对齐 YOPO_Sim)
        self._data['train'] = True
    
    def __getitem__(self, key: str) -> Any:
        """支持 cfg['env'] 方式访问"""
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any):
        """支持 cfg['env'] = value 方式设置"""
        self._data[key] = value
    
    def get(self, *keys, default=None):
        """
        支持嵌套访问 cfg.get('env', 'map_size')
        也支持单键访问 cfg.get('train', default=True)
        """
        data = self._data
        try:
            for key in keys:
                if isinstance(data, dict):
                    data = data[key]
                else:
                    return default
            return data
        except (KeyError, TypeError):
            return default
    
    def reload(self):
        """重新加载配置"""
        self._load_config()
    
    def set_test_mode(self, velocity: float = None):
        """
        切换到测试模式 (对齐 YOPO_Sim)
        
        Args:
            velocity: 测试时的速度，若为 None 则使用配置中的 max_vel
        """
        self._data['train'] = False
        robot = self._data['robot']
        train = self._data['training']
        traj = self._data['trajectory']
        
        vel_max_train = float(train.get('vel_max_train', robot['max_vel']))
        test_vel = velocity or robot['max_vel']
        
        # 速度缩放比例 (对齐 YOPO_Sim LatticeParam)
        ratio = test_vel / vel_max_train
        
        # 更新测试时的参数
        self._data['velocity'] = test_vel
        self._data['vel_max'] = ratio * vel_max_train
        self._data['acc_max'] = ratio * ratio * float(train.get('acc_max_train', robot['max_acc']))
        traj['segment_time'] = 2.0 * traj['radio_range'] / vel_max_train / ratio


# 全局配置实例
cfg = Config()
