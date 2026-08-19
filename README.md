# BasketEditor

BasketEditor 是一个基于 SAM2 的篮球视频进攻片段提取工具。

用户通过网页界面标注主人公、篮筐和篮球后，工具会追踪三个目标，对篮球轨迹进行短缺失补帧和平滑处理，并根据“主人公控球 → 发动进攻 → 篮球碰框”的时序规则识别有效进攻，最后自动导出对应的视频片段。

项目不区分投篮是否命中，也不需要训练额外的事件识别模型。

https://github.com/ChrisWang-jh/basketball_editor/assets/demo.mp4

## Quick Start

### 1. 准备环境

建议使用 Python 3.10 或更高版本，并安装基础依赖：

```bash
conda create -n basketedit python=3.10
conda activate basketedit
pip install numpy opencv-python gradio torch torchvision

mkdir {checkpoints, third_party}
```

此外还需要：

- 安装 FFmpeg，并确保终端可以执行 `ffmpeg`。
- 准备官方 SAM2 仓库。本项目已将其放在 `third_party/sam2` 时，可以直接使用该路径。
- 将匹配的 SAM2 权重放在 `checkpoints/`。默认权重文件为 `checkpoints/sam2.1_hiera_base_plus.pt`。

### 2. 启动剪视频ui

把包含原始视频的文件夹传给启动脚本。程序会递归查找其中的
`.mp4`、`.mov`、`.mkv`、`.avi`、`.webm` 等视频：

```bash
python run.py /path/to/source/videos --sam2-repo third_party/sam2
```

如需指定设备：

```bash
python run.py /path/to/source/videos \
  --sam2-repo third_party/sam2 \
  --device cuda
```

程序启动后，打开终端显示的 Gradio 地址。

### 3. 标注与分析

1. 用顶部的下拉框或左右按钮切换视频；每个视频的标注会分别保留。
2. 选择视频帧以及 `player`、`hoop` 或 `ball`。
3. 点击画面添加前景点；追踪到错误区域时可添加排除点。
4. 主人公和篮筐至少各添加一个前景点。
5. 建议在 3～10 个清晰帧中重复标注篮球。
6. 点击 **Start tracking and clipping** 开始分析当前视频。

结果会写入：

```text
outputs/<视频名称_分析时间>/
```

其中包括轨迹数据、事件数据、调试视频以及识别出的进攻片段。调试视频使用
浏览器兼容的 H.264 编码，并持续显示主人公、篮筐和篮球的 mask、轮廓、名称、
bounding box 及当前追踪状态。

### 4. 启动拼视频ui

启动独立的拼接界面（默认递归搜索项目的 `outputs/`）：

```bash
python stitch_ui.py
```

也可以指定其他输出目录和端口：

```bash
python stitch_ui.py /path/to/outputs --port 16667
```

界面只收集文件名以 `attack` 开头的视频。用左右按钮浏览，点击
**Select current** 按最终顺序选择，再点击 **Stitch selected clips**。程序会先统一
分辨率、帧率和音频格式，再输出 `stitched_<时间>.mp4` 到所选输出目录。

## 项目结构

```text
run.py                    命令行入口
basketeditor/ui.py        Gradio 标注界面
basketeditor/sam2.py      SAM2 预览与视频追踪
basketeditor/rules.py     补帧、平滑与进攻识别规则
basketeditor/pipeline.py  分析流程编排
basketeditor/video.py     视频读写与结果导出
basketeditor/models.py    数据模型和规则参数
stitch_ui.py              attack 片段拼接界面入口
basketeditor/stitch.py    视频标准化与拼接
```
