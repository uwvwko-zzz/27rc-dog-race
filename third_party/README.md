# 内置机器人与策略来源

以下文件来自本项目维护者提供的 `unitree_rl_mjlab` 本地 Go2-W 扩展工程，上游项目为 [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)。

| 内容 | 仓库位置 |
|---|---|
| Go2-W 模型 | `src/assets/robots/unitree_go2w/xmls/go2w.xml` |
| 模型引用的 17 个 mesh | `src/assets/robots/unitree_go2w/xmls/assets/` |
| ONNX 行走策略 | `artifacts/go2w_good/policy.onnx` |
| 策略部署实现 | `scripts/go2w_mujoco_validate.py` |

复制时逐文件校验了 SHA-256，与提供的原文件一致。模型、策略和部署实现按匹配的组合保留。

原工程根目录附带的 Apache-2.0 许可文本原样保存在 [unitree_rl_mjlab.LICENCE](unitree_rl_mjlab.LICENCE)。该文本保留原工程的许可信息；本项目原创场地与其他第三方素材的来源另见主 README。

Go2-W 模型、部署脚本和策略是所提供工程的本地扩展文件，不应视为官方上游仓库原始发行内容。内置策略用于行走演示，未验证通过全部比赛障碍或执行物资搬运任务。

浏览器控制依赖 [mujoco-gui](https://github.com/uwvwko-zzz/mujoco-gui)，通过 `requirements-sim2sim.txt` 安装；未将该依赖的源码复制到此目录。
