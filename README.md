# 2027 ROBOCON 足式机器人挑战赛 · MuJoCo 场地

根据《2027 年足式机器人挑战赛比赛规则》搭建的 **障碍赛与任务赛 MJCF 场景**，用于场地预览、机器人越障测试和 sim-to-sim 验证。另提供 Go2-W ONNX 行走策略的部署入口，可在浏览器面板中切换两张地图、控制机器人并实时调参。

障碍赛已按 V2.0（2026-10-29）更新：梅花桩及两端平台高度为 100 mm，轮胎为黑色。任务赛场地沿用 V1.0 建模，当前未实现自动判分。

场地按规则尺寸和示意图建模，是训练与验证用实现；障碍赛最终摆放位置以赛事公布为准。

![Go2-W 浏览器仿真面板：障碍赛地图](docs/images/gui-obstacle.png)

*实际浏览器界面：左侧为 MuJoCo 画面，下方预留菜单用于选择地图和视角，右侧可调整质量、载荷、摩擦和重力。*

## 更新日志

### 2026-10-29

- 按比赛规则 V2.0 更新障碍赛地图：梅花桩方块由 200 × 200 × 200 mm 改为 200 × 200 × 100 mm，两端平台由 800 × 800 × 200 mm 改为 800 × 800 × 100 mm，底面保持贴地。
- 核对轮胎为规则要求的黑色，同步更新障碍赛生成脚本、场地 XML 与建模说明。
- 重新渲染障碍赛斜视图、俯视图，并更新浏览器面板截图。
- 通过几何尺寸核对、MuJoCo 3.11.0 加载与 100 步运行验证。

## 地图预览

| 障碍赛 | 任务赛 |
|---|---|
| [![障碍赛：八类障碍的整体布局](docs/images/obstacle-overview.png)](docs/images/obstacle-overview.png) | [![任务赛：物资箱、归位区和挡车杆](docs/images/task-overview.png)](docs/images/task-overview.png) |
| 14 × 6 m 场地，包含绕桩、限高框架、高墙、沟壑组合、砖木坑、轮胎阵、梅花桩和单木桥。 | 6 × 4 m 内场，包含 8 个动态物资箱、8 个存放区、4 个归位区和 2 道挡车杆。 |

更多图片、俯视图与箱体贴图特写见 [图片介绍](docs/gallery.md)。

## 功能

- 两套场地 XML 和对应生成脚本，可单独加载场地，也可接入自己的机器人。
- 轮胎保留孔洞；砖木坑具有固定起伏；沟壑包含台阶、顶部平台和 20° 斜坡。
- 物资箱可独立运动，六面使用规则 PDF 中的标识；支持后续搬运、接触和堆叠控制器接入。
- Go2-W 演示支持浏览器 GUI、原生 MuJoCo 窗口和无窗口运行。
- GUI 中实时切换障碍赛 / 任务赛，自动重置机器人、策略历史和物资箱。
- 支持速度控制、鼠标视角操作、推力测试和控制参数调整。

当前策略负责行走；本项目尚未实现自主路线规划、物资识别与抓取、任务搬运或自动判分。

## 先体验场地

**仅查看地图不需要机器人模型或 ONNX 策略。** 推荐 Python 3.10，原生查看器需要可用的桌面和 OpenGL 环境。

```bash
git clone https://github.com/uwvwko-zzz/27rc-dog-race.git
cd 27rc-dog-race

conda create -n robocon-mujoco python=3.10 -y
conda activate robocon-mujoco
python -m pip install -r requirements.txt

# 障碍赛
python scripts/view_course.py

# 任务赛
python scripts/view_course.py --map task
```

直接在自己的 MuJoCo 程序中加载：

```python
import mujoco

model = mujoco.MjModel.from_xml_path("maps/task_course_2027.xml")
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
```

障碍赛 XML 自包含。任务赛 XML 使用 `assets/task_labels/` 中的纹理，复制场地时请保留 `maps/` 与 `assets/` 的相对目录位置。

## Go2-W 策略演示

### 安装并启动

仓库已包含 Go2-W 模型、引用的 mesh、部署实现和示例 ONNX 策略，无需另行准备 `unitree_rl_mjlab` 工程：

```text
scripts/go2w_mujoco_validate.py               # 策略部署实现
src/assets/robots/unitree_go2w/xmls/
├── go2w.xml                                # Go2-W 模型
└── assets/                                 # 模型引用的 17 个 mesh
artifacts/go2w_good/policy.onnx              # 示例行走策略
```

在项目根目录安装演示依赖并启动，其中浏览器面板来自 [mujoco-gui](https://github.com/uwvwko-zzz/mujoco-gui)：

```bash
python -m pip install -r requirements-sim2sim.txt
python sim2sim_course.py --gui
```

默认模型与权重均从当前项目读取。示例策略输入形状为 `[1, 275]`，输出为 `[1, 16]`，对应 12 个腿关节和 4 个轮关节。

如果希望使用自己的兼容工程或策略，可覆盖路径：

```bash
python sim2sim_course.py --gui --policy /path/to/policy.onnx
python sim2sim_course.py --gui --project /path/to/compatible_project
```

`--project` 工程应具备相同的部署代码与模型目录结构；未指定 `--policy` 时，读取该工程内的 `artifacts/go2w_good/policy.onnx`。也可用 `GO2W_PROJECT` 环境变量指定默认工程；不设置该变量即可使用内置资源。

浏览器默认打开 **http://127.0.0.1:8765**。在画面下方的地图菜单选择“2027 障碍赛场地”或“2027 任务赛场地”，无需重启。视角支持机器人跟随、全场斜视、全场俯视以及鼠标自由操作。

### 控制方式

| 操作 | 浏览器 GUI | 原生窗口 |
|---|---|---|
| 前后 / 横移 / 转向 | 按住 `W/S`、`A/D`、`Q/E` | 相同 |
| 停止行走指令 | 松开按键、切换窗口，或点“急停” | 松开按键、切换窗口，或按 `X` |
| 重置 | 点“重置机器人” | `R` |
| 推力测试 | 点“随机推力” | `P` |
| 暂停 | — | 空格 |
| 移动相机 | 左键旋转、右键平移、滚轮缩放 | MuJoCo 查看器的相机操作 |

默认指令幅值为前后 / 横移 **0.5 m/s**、转向 **1.0 rad/s**。交互模式的 `--vx`、`--vy`、`--yaw` 设置幅值，最低值与参考 GUI 保持一致；无窗口模式则持续执行指定速度指令。松键停止表示速度指令归零，机器人仍由策略维持姿态。

原生窗口的松键与失焦检测使用 Linux X11 和 `python-xlib`；其他桌面环境可优先使用浏览器 GUI。

### 常用命令

下列命令均在项目根目录运行，默认使用内置模型和策略。

```bash
# 双地图 GUI，默认先打开障碍赛
python sim2sim_course.py --gui

# 默认先打开任务赛，仍可在菜单中切换
python sim2sim_course.py --gui --map task

# 障碍赛从斜坡侧启动区 2 出发
python sim2sim_course.py --gui --start 2

# 原生窗口：任务赛
python sim2sim_course.py --map task

# 无窗口运行 5 秒仿真时间
python sim2sim_course.py --headless --duration 5 --vx 0.5

# 调低渲染负载，或更换端口
python sim2sim_course.py --gui --gui-fps 30 --gui-port 8766

# 服务器上启动服务，由自己打开浏览器
python sim2sim_course.py --gui --no-open-browser

# 自定义出生位置 X Y Z 与朝向 YAW（米、弧度）
python sim2sim_course.py --spawn 0 0 0.4 1.5708
```

更多参数见 `python sim2sim_course.py --help`。无窗口截图使用 `MUJOCO_GL=egl` 配合 `--snapshot output.png`，需要可用的 EGL 环境。

GUI 默认目标帧率为 **60 FPS**，画面为 **1280 × 960、JPEG 质量 92**；目标帧率不等于设备实测帧率。默认关闭投影阴影以减少组合场地的条纹和噪点，保留方向光照。需要阴影时加 `--gui-shadows`。物理步长为 **0.005 s**，策略频率为 **50 Hz**，独立于画面目标帧率。

## 目录结构

```text
.
├── README.md
├── sim2sim_course.py                 # 保持原来的部署启动命令
├── requirements.txt
├── requirements-sim2sim.txt
├── maps/                            # 可直接加载的两套 MJCF 场地
├── scripts/                         # 场地生成、地图预览、策略部署实现
├── assets/task_labels/              # 物资箱标识纹理
├── artifacts/go2w_good/policy.onnx   # 内置示例策略
├── src/assets/robots/unitree_go2w/   # 内置机器人模型与 mesh
├── docs/                            # 图片介绍、建模说明、正式展示图片
│   └── reference/                   # 本地规则 PDF，不纳入默认发布
├── third_party/                     # 来源说明与第三方许可
└── generated/                       # 运行时生成，自动创建，不纳入提交
```

## 文件与地图调整

| 文件 | 用途 |
|---|---|
| `maps/obstacle_course_2027.xml` | 纯障碍赛场地，可直接加载 |
| `maps/task_course_2027.xml` | 纯任务赛场地，依赖标识纹理 |
| `scripts/generate_course.py` | 重新生成障碍赛布局 |
| `scripts/generate_task_course.py` | 重新生成任务赛布局及箱子顺序 |
| `scripts/view_course.py` | 无需策略的地图预览入口 |
| `sim2sim_course.py` | Go2-W 原生 / GUI / 无窗口演示 |
| `scripts/go2w_mujoco_validate.py` | 内置策略部署实现 |
| `src/assets/robots/unitree_go2w/xmls/` | Go2-W 模型与 mesh |
| `artifacts/go2w_good/policy.onnx` | 内置示例行走策略 |
| `third_party/` | 模型来源与第三方许可文本 |
| `assets/task_labels/` | 食品、工具、仪器、药品标识 |
| `docs/modeling.md` | 尺寸、坐标、建模假设和机器人接入说明 |
| `docs/gallery.md`、`docs/images/` | 图片介绍与展示图片 |

修改对应生成器后运行：

```bash
python scripts/generate_course.py
python scripts/generate_task_course.py
```

障碍赛可修改各个障碍 body 的 `pos` 调整布局。任务赛可修改生成器中的 `permutations` 调整两行箱子的类别顺序。坐标与详细尺寸见 [建模说明](docs/modeling.md)。

部署时生成的 `generated/go2w_native.xml`、`generated/go2w_course.xml`、`generated/go2w_task_course.xml` 和 `generated/go2w_maps.xml` 含当前项目位置的绝对资源路径，可重新生成，不作为可移植场地文件分发。GUI 的双地图 XML 由部署脚本负责地图激活，单独预览请加载纯场地 XML。

## 验证范围

已在 MuJoCo **3.2.3** 和 **3.11.0** 中加载场地；Go2-W 演示在 Linux X11、MuJoCo 3.2.3 环境中验证了基本行走、两个启动区、按键释放、物资箱稳定性与重置，以及 GUI 双地图切换、实时画面和调参。

这些检查验证场地和部署链路可用，尚未证明策略可以通过全部比赛障碍或完成任务赛搬运。几何尺寸按规则建模，接触与材料参数需要结合实物校准。

## 来源与许可

场地依据《2027 年足式机器人挑战赛比赛规则》，障碍赛按 V2.0、任务赛按 V1.0 建模。障碍赛对应第 3.1 节、图 1-1 / 1-2、表 1，任务赛对应第 3.2 节、图 2–6。物资箱标识图片直接提取自规则 PDF 图 3。规则原文、赛事标识和这些图片属于第三方来源素材。

内置机器人、mesh、策略及部署实现取自 [Unitree RL mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) 的本地 Go2-W 扩展工程。原工程附带的 Apache-2.0 许可文本保存在 [third_party/unitree_rl_mjlab.LICENCE](third_party/unitree_rl_mjlab.LICENCE)，资源清单见 [来源说明](third_party/README.md)。浏览器控制面板使用 [mujoco-gui](https://github.com/uwvwko-zzz/mujoco-gui)，物理仿真使用 [MuJoCo](https://github.com/google-deepmind/mujoco)。这些依赖及素材各自的许可不由本项目替代。

本仓库尚未指定原创代码的开源许可证。
