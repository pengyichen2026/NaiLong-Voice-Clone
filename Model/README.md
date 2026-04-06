---
license: mit
datasets:
- pengyichen/NaiLong-Voice-Clone
---

# 奶龙语音克隆模型 (NaiLong Voice Clone Model)

[使用 app.py 得到的 Demo](https://dub.sh/nailong)

本仓库提供了：

1. 奶龙语音克隆模型的权重。
2. 高性能通用推理应用 (`app.py`)：
   将 `app.py` 放入安装好的 GPT-SoVITS 文件夹中，即可对于**通用**的特定角色声音克隆任务，只要有微调后的模型权重，即可立即生成一个支持**超快速推理**与**超低延迟流式输出**的网页。  
   - “**超快速推理**” 在仅仅一个 4090 上即可在 15s 内生成大约 5 分钟的长音频，3 ~ 5s 生成大约 20s 的中等长度音频。  
   - “**超低延迟流失输出**” 则可以在一个 4090 上，在 2 ~ 3s 的等待之后对任意长文本进行连续说话。  
   这既解决了**通用特定角色声音克隆任务**的模型权重到网页部署的过程，又解决了大部分语音克隆软件不支持流式输出以及推理速度慢的问题。
3. GPT-SoVITS 的一些补丁，解决了整型溢出问题并优化了兼容性。

## 模型权重

### 下载

由于 `GPT_weights` 与 `SoVITS_weights` 模型权重体积过大，GitHub 仅保留代码与 `reference.wav`，即推理时使用的参考音频。

完整仓库有如下两种下载方式：

1. Hugging Face (推荐)： [huggingface ](https://huggingface.co/pengyichen/NaiLong-Voice-Clone/tree/main)下载
2. HF-Mirror (国内镜像)：[hf-mirror ](https://hf-mirror.com/pengyichen/NaiLong-Voice-Clone/tree/main)下载

### 说明

我们基于 GPT-SoVITS 的 v2proplus 预训练模型，并使用 [奶龙语音克隆数据集](https://huggingface.co/datasets/pengyichen/NaiLong-Voice-Clone) 中的 `nailong_selected` 音频集进行微调，得到了最终的奶龙语音克隆模型。

`GPT_weights` 与 `SoVITS_weights` 中存放了我们通过多次实验，得到的最优的一组 GPT 模型权重与 SoVITS 模型权重，具体选择其中的哪一个可以根据你的需求决定。

## 高性能通用推理应用

### 功能

我们提供 `app.py`，有如下功能：

- “生成音频”，即快速推理，可以支持在网页上播放、暂停、滑动、下载等。
- “开始说话”，即流式输出，可以支持任意时刻暂停，以及重新 “开始说话”。
- 全程使用并行推理，至多支持同时有 4 个用户在流式输出中。
- 所有用户的输入文本与 “生成音频” 得到的音频会存储在 `generated_audios` 文件夹中，并按照时间与内容建立文件夹，方便直观得知用户访问时间密度分布与生成内容兴趣等。
- 可以支持多种语言：如果模式为 `auto` 即可以支持中，英，日，韩，粤 5 种语言。

### 安装

1. 安装 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)。
2. 将 `GPT_weights`、`SoVITS_weights`、`app.py`、`reference.wav` 下载，并放入 `GPT-SoVITS` 文件夹当中。
3. 运行 `app.py`，即可生成一个本地的 `6006` 号端口作为网页了。

## GPT-SoVITS 补丁

1. `2-get-hubert-wav32k.py`：修复了 GPT-SoVITS 对应代码中的 Integer Overflow 问题。

2. `TTS.py` 与 `2-get-sv.py`：针对 `torchaudio.load()` 进行了等价修改，解决了在特定 CUDA 环境下 `torchcodec` 的兼容性冲突。