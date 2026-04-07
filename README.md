---
license: mit
---

如果这个项目仓库对你有帮助，欢迎点个 Star ⭐ 支持一下！这是对我持续更新的最大鼓励！

[Demo](https://dub.sh/nailong)

# 奶龙语音克隆模型（Model）

模型仓库提供了：

1. 奶龙语音克隆模型的权重。
2. 高性能通用推理应用 (`app.py`)：
   将 `app.py` 放入安装好的 GPT-SoVITS 文件夹后，对于**通用**的特定角色声音克隆任务，只要有微调后的模型权重，即可立即生成一个支持**超快速推理**与**超低延迟流式输出**的网页。  
   “**超快速推理**”：在单个 4090 上推理，即可在 15s 内生成大约 5 分钟的长音频，3 ~ 5s 生成大约 20s 的中等长度音频。  
   “**超低延迟流失输出**”：在单个 4090 上流式输出，可以在 2 ~ 3s 的等待之后对任意长文本进行连续说话。  
   既为**通用特定角色声音克隆任务**打通了从“模型权重”到“前端交互”的最后一步，又解决了大部分语音克隆软件不支持流式输出以及推理速度慢的问题。
3. GPT-SoVITS 的一些补丁，解决了整型溢出问题并优化了兼容性。

## 模型权重

### 下载

由于 `GPT_weights` 与 `SoVITS_weights` 模型权重体积过大，GitHub 仅保留代码与参考音频 `reference.wav`。

完整模型仓库有如下两种下载方式：

1. Hugging Face (推荐)： [huggingface ](https://huggingface.co/pengyichen/NaiLong-Voice-Clone/tree/main)下载
2. HF-Mirror (国内镜像)：[hf-mirror ](https://hf-mirror.com/pengyichen/NaiLong-Voice-Clone/tree/main)下载

### 说明

我们基于 GPT-SoVITS 的 v2proplus 预训练模型，并使用 [奶龙语音克隆数据集](https://huggingface.co/datasets/pengyichen/NaiLong-Voice-Clone) 中的 `nailong_selected` 音频集进行微调，得到了最终的奶龙语音克隆模型。

`GPT_weights` 与 `SoVITS_weights` 中存放了我们通过多次实验，得到的最优的一组 GPT 模型权重与 SoVITS 模型权重，具体选择其中的哪一个可以根据你的需求决定。

## 高性能通用推理应用

### 功能

我们提供 `app.py`，有如下功能：

- “生成音频”，即快速推理，可以支持在网页上播放、暂停、滑动、倍速、下载生成好的音频。
- “开始说话”，即流式输出，可以支持任意时刻暂停，以及重新 “开始说话”。
- 使用并行推理进行加速，至多支持同时有 4 个用户在流式输出中。
- 所有用户使用 “生成音频” 时，对应的输入文本与得到的音频会存储在 `generated_audios` 文件夹的一个子文件夹当中，其中子文件夹的名称包含生成时间与输入文本内容，方便直观得知用户访问时间密度分布与生成内容兴趣等。
- 可以支持多种语言：将 `text_lang` 设为 `auto` 即可以支持中，英，日，韩，粤 5 种语言。

### 安装

1. 安装 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)。
2. 将 `GPT_weights`、`SoVITS_weights`、`app.py`、`reference.wav` 下载，并放入 `GPT-SoVITS` 文件夹当中。
3. 运行 `app.py`，即可生成一个本地的 `6006` 号端口作为网页了。

## GPT-SoVITS 补丁

1. `2-get-hubert-wav32k.py`：修复了 GPT-SoVITS 对应代码中的 Integer Overflow 问题。
2. `TTS.py` 与 `2-get-sv.py`：针对 `torchaudio.load()` 进行了等价修改，解决了在特定 CUDA 环境下 `torchcodec` 的兼容性冲突。

# 奶龙语音克隆数据集（Datasets）

## 数据集下载

由于 `raw_audio`、`vocal_only`、`sliced_vocal` 数据集体积过大，GitHub 仅保留代码与 `nailong_selected` 数据集。

完整数据集仓库有如下两种下载方式：

1. Hugging Face (推荐)： [huggingface ](https://huggingface.co/datasets/pengyichen/NaiLong-Voice-Clone/tree/main)下载
2. HF-Mirror (国内镜像)：[hf-mirror ](https://hf-mirror.com/datasets/pengyichen/NaiLong-Voice-Clone/tree/main)下载

## 数据集介绍

数据集按处理阶段分为以下四部分：

### 1. `raw_audio` (原始采样)
- **处理方式**：使用 Audacity 直接对视频素材进行录音，格式为 44.1kHz, 16-bit, Stereo。
- **说明**：包含背景音、特效及多角色对话的非结构化原片素材，是整个流水线的起点。

### 2. `vocal_only` (人声分离)
- **处理方式**：从 `raw_audio` 中使用 UVR5 的 MDX-Net 模型剥离背景音乐与噪音。
- **说明**：利用 MDX-Net 模型提取出干净的人声轨道，为后续切片提供高信噪比素材。

### 3. `sliced_vocal` (自动化切片)
- **处理方式**：基于停顿检测、音色突变及总时长控制，将 `vocal_only` 自动化切分为一系列短音频。
- **说明**：主要意义是为了后续通过选择器筛选出其中是 “纯正奶龙” 的片段。

### 4. `nailong_selected` (精选集)

该数据集可最终用于模型训练与微调，其由两部分构成：

* **人工精选部分**：从 `vocal_only` 中手动挑选的优质奶龙音色参考音频（44.1kHz, Stereo）。
* **选择器补充部分**：利用选择器 `selector.py`，以第一部分得到的奶龙参考音频为基础，通过迭代标记扩散，从 `sliced_vocal` 中检索出高置信度片段，并经人工二次核验挑选得到最终音频集（32kHz, Mono）。

## 音频选择器介绍

本仓库还提供了一份**通用**的音频筛选工具 `selector.py`。

该工具根据 `reference` 文件夹当中存放的特定角色参考音频，将 `sliced_vocal` 当中 “大概率是纯该特定角色” 的音频放入 `preselected` 文件夹中。

### 详细功能
* **全路径提取**：提取 `reference` 和 `sliced_vocal` 内部的任意层级子文件夹。
* **带路径输出**：在输出结果时，会在文件名中体现其在 `sliced_vocal` 目录当中的相对路径。这不仅能有效防止同名文件冲突，也极大方便了后期的数据溯源。
* **兼容多种格式**：支持 `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.opus` 等多种主流音频格式。

### 使用方法

1. **安装依赖**：
   ```
   pip install -r requirements.txt
   ```

2. **环境要求**：如需处理 `.mp3`, `.m4a`, `.opus` 等压缩格式，请确保系统已安装 FFmpeg。

3. **放置数据**：将参考音频放入 `reference`，待筛选切片放入 `sliced_vocal`。

4. **运行程序**：运行 `selector.py`，结果将输出至 `preselected` 文件夹。

## 个性化配置音频选择器说明

你可以根据待处理数据集质量和角色音色辨识度等情况，在 `selector.py` 顶部的 “配置区” 进行参数调整。

`selector.py` 的 “配置区” 中所有参数的详细含义及默认值如下：

```py
# ================= 配置区 ================= 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_DIR = os.path.join(BASE_DIR, "reference")      # 存放 “特定角色” 的参考音频
POOL_DIR = os.path.join(BASE_DIR, "sliced_vocal")   # 存放切割好的数据集
OUTPUT_DIR = os.path.join(BASE_DIR, "preselected")  # 存放程序最后选好的 “疑似该角色” 的片段

# 每一轮选择相似度 >= 对应轮 THRESHOLDS 的片段加入 “疑似该特定角色” 当中，并按照一定的衰减比例（DECAY_FACTOR）更新目标音色特征向量。
THRESHOLDS = [0.90,0.89,0.88,0.87,0.865,0.86,0.857,0.854,0.852,0.85]

# 待处理音频后缀，可按需增删。注意：处理压缩格式 (.mp3/.m4a/.opus 等) 需预装 FFmpeg ！
AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus')

# 衰减比例；初始就在 reference 数据集当中的每个音频对 “目标音色” 的贡献权重都为 1，后续第 i 轮（“轮数” 从 1 开始标号）加入的每个音频的贡献权重均为 DECAY_FACTOR ^ i。
DECAY_FACTOR = 0.8
# ========================================== 
```
