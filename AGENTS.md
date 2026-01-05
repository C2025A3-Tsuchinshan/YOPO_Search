# Repository Guidelines

## 项目结构与模块组织
本仓库包含两个子工程：`YOPO_Sim/` 为 3D 版本（ROS + CUDA + Python），`YOPO_2D/` 为纯 Python 的 2D 简化版。核心算法与训练代码在 `YOPO_Sim/YOPO/`（policy、loss、config），仿真与传感器在 `YOPO_Sim/Simulator/`，控制与 ROS 包在 `YOPO_Sim/Controller/`。2D 版本源码集中在 `YOPO_2D/`，配置为 `YOPO_2D/config/config.yaml`，训练与测试入口分别是 `YOPO_2D/train.py` 和 `YOPO_2D/test.py`。示例资源与说明文档通常位于各子工程的 `docs/` 或 `readme.md`。

## 构建、测试与开发命令
- 环境与依赖：`conda create -n yopo python=3.8`，然后 `pip install -r YOPO_Sim/YOPO/requirements.txt` 或 `pip install -r YOPO_2D/requirements.txt`。
- 3D 仿真构建：`cd YOPO_Sim/Controller && catkin_make`；`cd YOPO_Sim/Simulator && catkin_make`。
- 3D 联调：`roslaunch so3_quadrotor_simulator simulator_attitude_control.launch`，`rosrun sensor_simulator sensor_simulator_cuda`，最后 `python YOPO_Sim/YOPO/test_yopo_ros.py --trial=1 --epoch=50`。
- 2D 训练与测试：`python YOPO_2D/train.py`，`python YOPO_2D/test.py --no_vis`；日志可用 `tensorboard --logdir=YOPO_2D/saved/` 查看。

## 编码风格与命名约定
Python 代码使用 4 空格缩进，模块/函数用 `snake_case`，类名用 `CamelCase`，配置文件保持 YAML 风格并对齐注释。C++/CUDA 代码跟随现有格式（`YOPO_Sim/Simulator/src` 与 `YOPO_Sim/Controller/src`），不引入新的自动格式化工具，新增文件命名与同目录一致。

## 测试指南
仓库未发现专门的单元测试框架，主要以脚本级集成验证为主：3D 使用 `YOPO_Sim/YOPO/test_yopo_ros.py` 配合 ROS/仿真启动，2D 使用 `YOPO_2D/test.py`。若修改仿真或控制链路，建议完整跑一遍仿真流程并记录关键参数。

## 提交与合并请求
当前检出不含 Git 历史，无法总结既有提交规范。建议提交信息采用清晰动词开头并注明范围，例如 `feat(sim): add lidar noise model`。PR 请附简要描述、关键命令与配置变更说明；涉及可视化或行为变化时，附 RViz/2D 输出截图或短视频链接。

## 配置与产物提示
关键配置路径：`YOPO_Sim/Simulator/src/config/config.yaml`、`YOPO_Sim/YOPO/config/traj_opt.yaml`、`YOPO_2D/config/config.yaml`。训练/数据输出可能出现在 `YOPO_Sim/YOPO/saved/`、`YOPO_2D/saved/`、`YOPO_Sim/dataset/`，避免将大规模生成数据直接纳入版本库。
