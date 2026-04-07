---
license: cc-by-nc-sa-4.0
---

# 奶龙语音克隆数据集

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

除数据集外，本仓库还提供了一份**通用**的音频筛选工具 `selector.py`。

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
