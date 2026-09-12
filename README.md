# BasketEdit

篮球视频标注、目标身份追踪、进攻剪辑与集锦拼接工具。默认使用本地 **SAM2 分割 + DINOv3 外观记忆**：保留人工标注的目标特征，目标离场后持续搜索，重新确认身份后恢复追踪。调试视频显示所选目标的像素分割轮廓。

界面由 Python 中的 Gradio 组件生成，采用浅奶油色背景、细边框、圆角和轻阴影。第 1 页是“裁剪视频”，第 2 页是“拼接集锦”；顶部通过“上一页 / 下一页”切换并显示页码，翻页会保留标注、当前画面和拼接选择。

## 启动与环境

本机已安装 `basketedit` 环境：

```bash
conda activate basketedit
cd /data2/jiahe_wang/github_codebase/basketeditor
python run.py src
```

默认监听 `http://127.0.0.1:16666`。设备 `auto` 优先选择 CUDA，其次 MPS、CPU；可通过 `--device cuda:7` 指定设备。修改界面或追踪代码后，需要重启服务并刷新浏览器。

在其他机器重新安装时，所有 Python 包使用清华源：

```bash
conda create -n basketedit python=3.11 pip -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda activate basketedit
git submodule update --init --recursive
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
SAM2_BUILD_CUDA=0 python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-build-isolation -e third_party/sam2
python -m pip config --site set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

系统还需安装 FFmpeg 和 ffprobe。`requirements.txt` 包含 PyTorch 2.5.1、torchvision 0.20.1、Gradio 及 SAM2 推理依赖。`SAM2_BUILD_CUDA=0` 只跳过可选的 CUDA 连通域扩展，仍可使用 GPU 运行模型。DINOv3 直接加载本地官方骨干网络，无需安装其训练依赖。

模型默认路径如下；模型权重和原始视频不包含在 Git 中。

| 路径 | 用途 |
| --- | --- |
| `third_party/sam2` | 官方 SAM2 源码子模块 |
| `checkpoints/sam2.1_hiera_base_plus.pt` | SAM2.1 base+ 权重 |
| `third_party/dinov3` | 官方 DINOv3 源码子模块 |
| `checkpoints/dinov3_vits16.pth` | 本地 DINOv3 ViT-S/16 权重的软链接 |

SAM2 和 DINOv3 都通过 `git submodule update --init --recursive` 获取固定版本源码；DINOv3 当前固定在 `6876159a11b4df116f30f667f8c9888617df0751`。本机 DINOv3 权重链接至 `/data2/jiahe_wang/github_codebase/dual_feature_ot_gen/checkpoints/dino_v3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`。迁移机器时需单独准备权重，也可使用 `--dino-repo`、`--dino-checkpoint` 指定自己的路径；SAM2 对应参数为 `--sam2-repo`、`--checkpoint`、`--model-config`。

`python run.py src --no-reid` 启动仅使用 SAM2 的对照模式。该模式依靠人工提示传播，没有 DINO 全图身份检索。

## 远程访问

在个人电脑上建立 SSH 转发，再打开本机浏览器的 `http://127.0.0.1:16666`：

```bash
ssh -N -L 16666:127.0.0.1:16666 jiahe_wang@服务器地址
```

内网服务器需先通过内网、VPN 或跳板机实现 SSH 可达。服务端可以继续监听 `127.0.0.1`；局域网直接访问时使用 `--host 0.0.0.0`。`--port` 可修改端口，默认不会创建公开分享链接。

启动入口自动将 localhost、回环地址及绑定地址加入 `NO_PROXY` / `no_proxy`，保留已有外网代理，避免 Gradio 自身的 `/gradio_api/startup-events` 请求经代理转发后返回 502。

## 标注、追踪与结果

1. 打开界面即进入第 1 页“裁剪视频”。选择视频，在时间轴上找到清晰画面，为人物、篮球、篮圈分别添加前景点。右侧显示标记进度；三个目标可以标在不同帧。
2. 用排除点去掉误选区域。篮圈应标在真实圈口，避免把整块篮板当成篮圈；界面仅提供前景点和排除点。
3. 同一人物的正面、背面或明显姿态变化可分别补标；小球尽量在清晰帧补标。后出现的篮圈可以直接在出现帧标注。
4. “更多设置”默认收起。篮筐默认“自动稳定”；摇镜时可选择“移动机位”，固定机位才使用“固定机位”。
5. 三个目标标记齐全后，点击“开始裁剪”。完成后显示片段和追踪回放，“下载文件与详细记录”中提供报告与下载；点击“下一步：拼接 →”可继续制作集锦。

标注按视频自动保存到 `outputs/annotations/`，刷新页面后恢复；原视频发生变化时不套用旧标注。撤销、清除标记作用于当前目标的当前帧。关闭“更多设置”中的“显示即时分割预览”可降低交互等待时间。

追踪过程中：

- SAM2 独立分割人工标注帧，DINOv3 提取目标轮廓内的整体、局部及颜色特征。模型使用预训练权重，运行时不训练神经网络。
- 人工身份锚点保持不变；同帧其他独立物体提供固定负样本。自动外观更新有额外核验条件和数量上限。
- SAM2 逐帧传播后接受身份与几何核验。目标丢失或预测被拒绝时，清理时序记忆，使用全图检索、局部搜索和独立 SAM2 分割寻找候选；小目标使用局部高分辨率处理。
- 人物转身时，连续帧允许用双向光流、局部图像内容和轮廓重叠验证像素运动，避免仅因外观分数下降而切断正确轨迹。这条路径只支持已经连续跟踪的人物，不授权丢失后的重新识别，也不更新身份模板。
- 定期检查竞争候选，防止持续追踪时主动换成远处更相似的人。明确匹配可在重现帧恢复，证据不足保留缺失或待确认；这些身份否决不会被轨迹插值覆盖。
- 连续运动与离场后的重新识别使用不同条件。多帧确认只能帮助解决高外观相似度候选的排序歧义，不能把一直看错的队友升级成目标。小球额外使用去背景、旋转平均特征和形状检查；较弱的小球外观必须同时满足短时速度预测、颜色与尺度连续，不能据此跨长缺失恢复。
- 切镜清除运动与 SAM2 时序状态，身份锚点继续保留。图像缓存和时序记忆有上限，运行中的临时 JPEG 与 mask 文件仍需要与视频长度相应的磁盘空间。

“长期记忆”指同一次视频分析中，人工身份锚点不会因离场时间过长而过期。它不能保证从模糊像素、完全相同的球衣或未见过的视角中恢复唯一身份。当前每次分析从保存的标注重建身份库，并导出该次特征文件；遇到持续缺失，可按报告帧号增加清晰视角的标注。

设计参考了 [REMIND 的 DINOv3 特征与身份记忆思路](https://github.com/cvar-vision-dl/remind-reid-tracker)。本项目保留 SAM2 分割，未移植 REMIND 的完整多目标关联、室内邻居图及 YOLO 检测流程；也没有针对篮球球员训练新的 ReID 网络。

结果保存在 `outputs/<视频名_时间>/`：

| 文件 | 内容 |
| --- | --- |
| `annotations.json` | 本次分析使用的提示 |
| `identity_memory.npz` | DINO 模式的固定身份锚点、负样本及可信外观记录 |
| `tracks.json` | 原始和最终轨迹、可见性、来源、身份分数与核验原因 |
| `tracking_report.json` | 缺失区间、身份检索诊断、恢复记录和切镜位置 |
| `debug_tracking.mp4` | 半透明分割填色、轮廓和逐帧状态 |
| `events.json` | 进攻、疑似命中、裁剪边界与时间戳 |
| `attack_001.mp4` 等 | 保留音轨的进攻片段 |

身份相似度是匹配分数，不是正确概率。回放将分割置信度 `mask` 与身份分数 `id` 分开显示，只为当前帧已核验的观测绘制真实分割；插值、固定位置估计使用虚线提示，缺失或待确认目标不会显示伪造的分割区域。

进攻识别分别使用人物尺度判断持球、篮圈尺度判断到筐，允许短暂缺失和快速出手。`likely_made` 表示二维轨迹符合从上向下穿过圈口，`rim_contact` 表示到筐进攻，都需要回放确认；它们不等于裁判确认的得分。明显切镜会分隔事件。裁剪使用逐帧时间戳，规则窗口和调试视频使用平均 FPS，帧间隔变化很大的视频建议先转为固定帧率。

## 片段库与拼接

点击“下一页 →”进入第 2 页“拼接集锦”，即可预览 `outputs/` 中的历史 `attack*` 片段。点选左侧缩略图，预览后点击“加入集锦”；右侧“播放顺序”按实际导出顺序列出已选片段，点击其中一段即可上移、下移或移除。点击“生成集锦”后显示预览与下载按钮。“← 上一页”可随时返回裁剪，片段库为空时也提供返回入口。

新分析完成后自动刷新片段库，也可手动刷新。切换源视频、翻页或刷新片段库会保留当前会话中仍存在的选择及顺序；重新打开网页后，历史片段仍可读取，选择队列按会话保存。

独立入口仍可使用：

```bash
python stitch_ui.py /path/to/outputs --port 16667
```

拼接会统一尺寸、帧率和音频格式，兼容有声与无声片段，生成 `stitched_<时间>.mp4`。

## 验证

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-dev.txt
python -m pytest tests -q
ruff check basketeditor tests scripts run.py stitch_ui.py
```

真实权重验证使用同一个脚本。受控场景依赖本地 `assets/demo.mp4`，该素材不随代码提交；没有该素材时，使用 `--natural --source 自己的视频 --annotations 标注文件` 验证：

```bash
# 实际球员抠图构成的控制序列：离场12秒、同队干扰、异地重现和歧义
python scripts/verify_tracking.py --device cuda:7 --gap-frames 120

# 未修改的原视频：默认 assets/demo.mp4 的 8 号球员
python scripts/verify_tracking.py --natural --device cuda:7

# 使用完全相同的输入和标注，对照原始 SAM2 传播
python scripts/verify_tracking.py --natural --no-reid --device cuda:7

# 使用已经保存的真实视频标注；可加 --max-frames N 验证前 N 帧
python scripts/verify_tracking.py --natural --device cuda:7 \
  --source src/wjh_2.mp4 \
  --annotations outputs/annotations/2ed9ef8f61b2a1e473a5.json
```

控制序列检查恢复延迟、空帧误追、歧义拒绝和分割 IoU。自然视频模式输出轨迹统计、分割回放和均匀取帧拼图；可见帧数是模型决策，不能视为追踪准确率。两者的结果位于 `outputs/validation/`。

2026-09-12 本机验收（SAM2.1 base+、DINOv3 ViT-S/16，CUDA 6 / 7）：

| 验证 | 结果与范围 |
| --- | --- |
| 回归测试 | 160 项通过，Ruff 与 `git diff --check` 通过 |
| [离场恢复控制序列](outputs/validation/recovery_20260912_122526/recovery_report.json) | 目标离场 120 帧（12 秒），两次重现均在首个可见帧恢复；目标缺席时误追 0 帧，相同外观副本歧义接受 0 帧 |
| [原始示例视频](outputs/validation/natural_20260912_122223/visual_review.json) | 人工只标第 0 帧；逐帧目检确认第 0～123 帧的分割都是 8 号；第 124 帧剩余边缘身体被保守拒绝，此后没有切换到其他队员。这段检验连续追踪和出画，不检验自然重入 |
| [用户视频人物](outputs/validation/natural_20260912_123001/visual_review.json) | `wjh_2.mp4` 前 512 帧，使用原有第 0、60、392 帧三个人工提示；模型接受 248 帧，缺失 264 帧。第 151、276、385 帧三次重连经抽查均为标定的白衣 14 号，但之前部分缺失帧中他仍然可见，且第 385 帧受后续人工模板帮助；不能称为自然出画后首帧恢复，连续性仍明显不足 |
| [用户视频小球](outputs/validation/natural_20260912_123330/natural_report.json) | 原始第 494～520 帧，仅使用第 494 帧球提示；模型接受前 10 帧，随后 17 帧缺失。独立目检只能确认前 7 帧仍是球，第 501～503 帧疑似跟到手部遮挡，不能算成功恢复；再次露球仍未可靠找回 |

人物回放：[完整分割视频](outputs/validation/natural_20260912_122223/debug_tracking.mp4)、[关键帧拼图](outputs/validation/natural_20260912_122223/segmentation_contact_sheet.jpg)。控制序列的合成方式、掩码 IoU、轨迹和身份库均保存在对应目录，可复查。

**当前实现通过了长期记忆与受控恢复机制验证，但尚未达到用户原片中人物、球都能永久追踪、每次重现立即识别的验收要求。** 模糊小球、相似球衣、完全遮挡以及分割混入背景仍会导致证据不足或身份歧义，继续降低阈值可能变成误追。固定身份库不会过期，并不意味着视觉模型永不出错。

这是离线分析：212 帧示例视频推理约 130 秒，512 帧用户人物片段约 352 秒；“首帧恢复”指输出帧的位置，不代表实时处理延迟。测试期间每个推理任务额外使用约 1.5 GB 显存，现有训练任务继续运行。

浏览器验收覆盖左右翻页、首尾按钮状态、标注与片段选择保留、异常后重试、片段预览、排序、实际拼接和下载、空片段库和手机布局。脚本自动在临时目录生成测试视频，裁剪分析由测试数据模拟，不调用 GPU 模型；标注保存、片段发现和 FFmpeg 拼接使用真实实现，并核对导出的视频顺序与下载内容。可加 `--video 本地视频` 生成更直观的界面截图。本机已安装 Playwright Chromium，运行：

```bash
PLAYWRIGHT_BROWSERS_PATH=/data2/jiahe_wang/.cache/basketedit-browsers python scripts/verify_ui.py
```

浏览器二进制由 Playwright 官方分发；Python 依赖仍通过清华 PyPI 源安装。

## 代码位置

| 模块 | 职责 |
| --- | --- |
| `run.py`、`stitch_ui.py` | 命令行入口 |
| `basketeditor/ui.py`、`stitch_ui.py`、`theme.py` | Gradio 页面、片段库与样式 |
| `basketeditor/annotations.py` | 标注校验和持久化 |
| `basketeditor/sam2.py` | 官方模型加载、预览和 SAM2 对照模式 |
| `basketeditor/tracker.py`、`reid.py` | 身份核验、恢复控制与 DINO 特征记忆 |
| `basketeditor/tracking.py`、`rules.py` | 轨迹处理、篮圈稳定和进攻判定 |
| `basketeditor/pipeline.py`、`video.py`、`stitch.py` | 分析编排、视频读写和片段导出 |

`third_party/` 保存源码入口，`checkpoints/` 保存权重入口；`assets/` 是演示素材，`src/` 是用户视频，`outputs/` 是生成结果。验证脚本与测试分别位于 `scripts/`、`tests/`。

环境统一使用 `basketedit`。测试与 Ruff 缓存统一放在 `outputs/.cache/`，配置见 `pyproject.toml`。

## Git 提交范围

`.gitignore` 在根目录采用白名单，只纳入应用代码、入口脚本、依赖清单、配置、README 和测试。新增根目录的代码或文档目录时，需要将其加入白名单。

- `src/`、`assets/`、`outputs/`、`checkpoints/`、本地环境、缓存和日志留在本机；模型权重、视频和临时文件即使误放进源码目录也会被忽略。
- SAM2 和 DINOv3 只提交 `.gitmodules` 和子模块版本引用，不将第三方完整源码重复加入主仓库；两者的权重使用本地路径，不纳入 Git。
- `assets/demo.mp4` 已停止 Git 跟踪，本地文件保留。忽略规则作用于后续提交，不会删除已有提交中的历史文件。

提交前可用 `git add --dry-run .` 检查将纳入的文件；本地素材和运行结果无需上传。
