# BasketEditor

BasketEditor 是一个基于 SAM2 的篮球视频进攻片段提取工具。

用户通过网页界面标注主人公、篮筐和篮球后，程序会追踪三个目标，对篮球轨迹进行短缺失补帧和平滑处理，并根据“主人公控球 → 发动进攻 → 篮球碰框”的时序规则识别有效进攻，最后自动导出对应的视频片段。

项目不区分投篮是否命中，也不需要训练额外的事件识别模型。

## Quick Start

### 1. 准备环境

建议使用 Python 3.10 或更高版本，并安装基础依赖：

```bash
pip install numpy opencv-python gradio torch torchvision
```

此外还需要：

- 安装 FFmpeg，并确保终端可以执行 `ffmpeg`。
- 准备官方 SAM2 仓库。本项目已将其放在 `third_party/sam2` 时，可以直接使用该路径。
- 将匹配的 SAM2 权重放在 `checkpoints/`。默认权重文件为 `checkpoints/sam2.1_hiera_base_plus.pt`。

### 2. 启动界面

```bash
python run.py /path/to/source/video --sam2-repo third_party/sam2
```

如需指定设备：

```bash
python run.py /path/to/source/video \
  --sam2-repo third_party/sam2 \
  --device cuda
```

程序启动后，打开终端显示的 Gradio 地址。

### 3. 标注与分析

1. 选择视频帧以及 `player`、`hoop` 或 `ball`。
2. 点击画面添加前景点；追踪到错误区域时可添加排除点。
3. 主人公和篮筐至少各添加一个前景点。
4. 建议在 3～10 个清晰帧中重复标注篮球。
5. 点击 **Start tracking and clipping** 开始分析。

结果会写入：

```text
outputs/<视频名称_分析时间>/
```

其中包括轨迹数据、事件数据、调试视频以及识别出的进攻片段。

## 项目结构

```text
run.py                    命令行入口
basketeditor/ui.py        Gradio 标注界面
basketeditor/sam2.py      SAM2 预览与视频追踪
basketeditor/rules.py     补帧、平滑与进攻识别规则
basketeditor/pipeline.py  分析流程编排
basketeditor/video.py     视频读写与结果导出
basketeditor/models.py    数据模型和规则参数
```
