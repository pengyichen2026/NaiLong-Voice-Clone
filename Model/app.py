import os
import sys
import io
import re
import time
import base64
import threading
import numpy as np
import scipy.io.wavfile as wavfile
import gradio as gr
import torch
import soundfile as sf
import gc

current_dir = os.path.dirname(os.path.abspath(__file__))
save_root = os.path.join(current_dir, "generated_audios")
os.makedirs(save_root, exist_ok=True)

# --- 1. 初始化模型 (确保路径正确) ---
sys.path.append(os.path.abspath("GPT_SoVITS"))
from TTS_infer_pack.TTS import TTS, TTS_Config

vits_path = "SoVITS_weights/NaiLong_e14_s504.pth"
gpt_path = "GPT_weights/NaiLong-e16.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = True if device == "cuda" else False

tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_config.device = device
tts_config.is_half = is_half
tts_config.version = "v2ProPlus"
tts_config.t2s_weights_path = gpt_path
tts_config.vits_weights_path = vits_path

tts_pipeline = TTS(tts_config)

def weighted_length(text):
    chinese_len = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_len = len(text)
    non_chinese_len = total_len - chinese_len
    
    return chinese_len * 2 + non_chinese_len

# --- 2. 文本处理 ---
def preprocess_text(text):
    if not text:
        return ""
    
    if weighted_length(text) > 10000:
        raise ValueError("输入过长")

    text = re.sub(r'[“”"",;，；、;《》\[\]()~`]', '，', text).strip()
    return text

def split_to_list(text):
    """根据标点符号将长文本分割为短句列表"""
    print("Original Text = ", text)
    text = preprocess_text(text)
    print("PreProcessed_Text = ", text)
    parts = re.split(r'([，。！？；,.!?;])', text)
    sentences = []

    for i in range(0, len(parts)-1, 2):
        sent = parts[i] + parts[i+1]
        if sent.strip():
            print(sent.strip())
            sentences.append(sent.strip())

    # 处理结尾没有标点的情况
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
        
    return sentences

# --- 3. 核心功能函数 ---

def standard_inference(text):
    """静态生成模式：生成整段音频"""
    processed_text = preprocess_text(text)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    short_text = re.sub(r'[\\/:*?"<>|]', '', processed_text)[:5]
    folder_name = f"{timestamp}_{short_text}"
    folder_path = os.path.join(save_root, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    inputs = {
        "text": processed_text, 
        "text_lang": "zh", 
        "ref_audio_path": "reference.wav", 
        "prompt_lang": "zh", 
        "prompt_text": "啊，我才不要这样，好害羞啊。", 
        "top_k": 5, 
        "batch_size": 10, 
        "parallel_infer": True
    }

    full_audio = []
    sr = 32000
    for item in tts_pipeline.run(inputs):
        sr, audio_data = item[0], item[1]
        full_audio.append(audio_data)

    if full_audio:
        concatenated_audio = np.concatenate(full_audio)
        wav_path = os.path.join(folder_path, "audio.wav")
        wavfile.write(wav_path, sr, concatenated_audio)
        with open(os.path.join(folder_path, "content.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        return wav_path
    return None

def start_action(text, state):
    state.clear()
    print("[DEBUG] start_action 被调用 | text长度:", len(text) if text else 0)

    if not text or not text.strip():
        print("[DEBUG] 文本为空，重置")
        yield f"RESET|{time.time()}", state, gr.update(), gr.update()
        return

    state["is_running"] = False
    sentences = split_to_list(text)

    if not sentences:
        print("[DEBUG] 文本为空，重置")
        yield f"RESET|{time.time()}", state, gr.update(), gr.update()
        return

    new_state = {
        "sentences": sentences,
        "current_idx": 0,
        "is_running": True,
        "audio_segments": [],
        "sample_rate": 32000
    }

    print(f"[DEBUG] 开始流式生成")

    yield f"RESET|{time.time()}", new_state, gr.update(visible=False), gr.update(visible=True)

    try:
        yield from run_streaming_inference(new_state)
    except Exception as e:
        import traceback
        print("[ERROR] start_action 中 yield from 异常:")
        print(traceback.format_exc())
        yield f"RESET|{time.time()}", new_state, gr.update(visible=True), gr.update(visible=False)


def run_streaming_inference(state):
    sentences = state["sentences"]
    GROUP_SIZE = 4

    while state["current_idx"] < len(sentences) and state.get("is_running", True):
        idx = state["current_idx"]
        end_idx = min(idx + GROUP_SIZE, len(sentences))
        if(end_idx + GROUP_SIZE > len(sentences)):
            end_idx = len(sentences)

        batch_text = "，".join(sentences[idx:end_idx])
        
        inputs = {
            "text": batch_text, "text_lang": "zh", "ref_audio_path": "reference.wav",
            "prompt_lang": "zh", "prompt_text": "啊，我才不要这样，好害羞啊。",
            "top_k": 5, "batch_size": 2, "parallel_infer": True,
        }

        try:
            for item in tts_pipeline.run(inputs):
                if not state.get("is_running", True): break
                sr, audio_data = item[0], item[1]
                state["sample_rate"] = sr

                # 转为 Base64 传回给 JS 播放
                print("得到音频！！！")
                buffer = io.BytesIO()
                sf.write(buffer, audio_data, sr, format='ogg', subtype='vorbis')
                b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                # 更新索引到这组句子的末尾
                state["current_idx"] = end_idx
                yield b64_str, state, gr.update(), gr.update()
        except Exception as e:
            print(f"推理出错: {e}")
            break

    print("推理结束！！！")
    if state["current_idx"] >= len(sentences) and state.get("is_running", True):
        state["is_running"] = False
        yield "FINISH", state, gr.update(visible=False), gr.update(visible=True)
    else:
        print("出错！！！")

def pause_action(state):
    """暂停流式生成"""
    state["is_running"] = False
    return f"PAUSE_SIGNAL|{time.time()}", state, gr.update(visible=True), gr.update(visible=False)

def finalize_ui():
    """隐藏按钮被JS点击后，真正重置 UI 状态"""
    return gr.update(visible=True), gr.update(visible=False)

# --- 4. JS 逻辑与 CSS 样式 ---
js_code = """
async (b64Data) => {
    if (!b64Data)
        return;
    console.log("--- [Debug] playNext 启动 ---");
    const signal = b64Data.split('|')[0];
    if (signal === "RESET" || signal === "PAUSE_SIGNAL") {
        window.audioQueue = [];
        window.isPlaying = false;
        if (window.globalAudioCtx) {
            await window.globalAudioCtx.close();
            window.globalAudioCtx = null;
        }
        return;
    }
    if (!window.audioQueue)
        window.audioQueue = [];
    window.audioQueue.push(b64Data);
    if (window.isPlaying)
        return;
    
    console.log("--- [Debug] playNext 启动 ---");
    console.log("当前队列长度:", window.audioQueue.length);
    console.log("当前播放状态 (isPlaying):", window.isPlaying);
    const playNext = async () => {
        if (window.audioQueue.length === 0) {
            window.isPlaying = false;
            return;
        }
        window.isPlaying = true;
        const data = window.audioQueue.shift();
        console.log("取出数据类型:", data === "FINISH" ? "【信号信号: FINISH】" : "【音频数据】");
        if(data=="FINISH") {
            const btn = document.getElementById('hidden_finish_btn');
            if (btn)
                btn.click(); 
            window.isPlaying = false;
            return;
        }
        
        if (!window.globalAudioCtx) {
            window.globalAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        const ctx = window.globalAudioCtx;
        try {
            const response = await fetch(`data:audio/ogg;base64,${data}`);
            const arrayBuffer = await response.arrayBuffer();
            const buf = await ctx.decodeAudioData(arrayBuffer);
            const src = ctx.createBufferSource();
            src.buffer = buf;
            src.connect(ctx.destination);
            src.onended = playNext;
            src.start(0);
        } catch (e) {
            console.error("音频解码/播放错误:", e);
            window.isPlaying = false;
        }
    };
    playNext();
}
"""

custom_css = """
.text_center {
    text-align: center;
}
#hidden_finish_btn {
    display: none !important;
}
#data_box {
    display: none !important;
}
.main_container {
    width: 70% !important;
    margin: 0 auto !important;
}
.green_btn {
    background: #28a745 !important;
    color: white !important;
    border: none !important;
}
.green_btn:hover {
    background: #218838 !important;
    color: white !important;
}
#audio-box-container {
    width: 100% !important;
    max-width: 100% !important;
    overflow: hidden !important;
    padding-bottom: 10px !important;
}
/* 核心滚动容器设置 */
#audio-box-container .audio-container,
#audio-box-container .waveform-container,
#audio-box-container audio ~ div,
#audio-box-container > div > div {
    overflow-x: auto !important;
    overflow-y: hidden !important;
    scrollbar-width: thin; 
    scrollbar-color: transparent transparent;
    padding-bottom: 2px !important;
    margin-bottom: 2px !important;
}
/* Chrome/Edge 滚动条 - 默认透明细条 */
#audio-box-container .audio-container::-webkit-scrollbar,
#audio-box-container .waveform-container::-webkit-scrollbar,
#audio-box-container audio ~ div::-webkit-scrollbar,
#audio-box-container > div > div::-webkit-scrollbar {
    height: 6px !important;
    background: transparent !important;
}
#audio-box-container .audio-container::-webkit-scrollbar-thumb,
#audio-box-container .waveform-container::-webkit-scrollbar-thumb,
#audio-box-container audio ~ div::-webkit-scrollbar-thumb,
#audio-box-container > div > div::-webkit-scrollbar-thumb {
    background: transparent !important;
    border-radius: 10px !important;
    border: 2px solid transparent !important;
}
/* 改进：只在 hover 到波形容器本身时才显示滚动条（更精准） */
#audio-box-container .audio-container:hover::-webkit-scrollbar-thumb,
#audio-box-container .waveform-container:hover::-webkit-scrollbar-thumb,
#audio-box-container audio ~ div:hover::-webkit-scrollbar-thumb,
#audio-box-container > div > div:hover::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.28) !important;   /* 浅灰，接近 Firefox */
}
/* Firefox hover 加强 */
#audio-box-container .audio-container:hover,
#audio-box-container .waveform-container:hover {
    scrollbar-color: rgba(0, 0, 0, 0.28) transparent;
}
/* 原生 audio 控件保护 */
#audio-box-container audio {
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
}
/* waveform 容器额外保护 */
#audio-box-container .waveform-container {
    padding-bottom: 12px !important;
}
"""

# --- 5. Gradio 界面构建 ---
with gr.Blocks(title="奶龙语音生成器", css=custom_css) as demo:
    with gr.Column(elem_classes="main_container"):
        
        # 页面 Header HTML
        gr.HTML("""
            <div style="display:flex; justify-content:space-between; align-items:flex-start; padding: 10px 0; border: none !important; margin-bottom: 0px !important; box-shadow: none !important;">
                <div style="font-size:28px; font-weight:bold;">🐉 奶龙语音生成器</div>
                <div style="display:flex; flex-direction:column; align-items:flex-start; gap:4px;">
                    <div style="font-size:13px; color:#666; text-align:right;">觉得有用？</div>
                    <a href="https://github.com/pengyichen2026/NaiLong-Voice-Clone" target="_blank" style="display:inline-flex; align-items:center; gap:10px; padding:8px 16px; border-radius:8px; background:#0969da; color:white; text-decoration:none; font-size:14px; font-weight:600; transition: all 0.2s ease;" onmouseover="this.style.background='#0b5fcc'" onmouseout="this.style.background='#0969da'">
                        <svg height="16" width="16" viewBox="0 0 16 16" fill="white" style="flex-shrink:0;">
                            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                        </svg>
                        <span style="display:inline-flex; align-items:center; gap:2px; color:white;">
                            给个
                            <svg height="14" width="14" viewBox="0 0 16 16" fill="white">
                                <path d="M8 0l2.47 5.01L16 5.82l-4 3.9L13.18 16 8 13.27 2.82 16 4 9.72 0 5.82l5.53-.81L8 0z"/>
                            </svg>
                            Star
                        </span>
                    </a>
                </div>
            </div>
        """)

        # 核心组件
        data_transfer = gr.Textbox(visible=True, elem_id="data_box")
        input_text = gr.Textbox(label="想让奶龙说什么呢？", placeholder="在这里打字...", lines=4, max_lines=8)
    
        with gr.Column(elem_id="audio-box-container"):
            output_audio = gr.Audio(label="音频生成结果", buttons=["download"])

        with gr.Column():
            gen_btn = gr.Button("生成整段音频", variant="primary")
            with gr.Row():
                start_btn = gr.Button("让奶龙开始说话", visible=True, elem_classes="green_btn")
                pause_btn = gr.Button("让奶龙终止说话", variant="stop", visible=False)

        state = gr.State({})

        # 1. 静态生成
        gen_btn.click(fn=standard_inference, inputs=input_text, outputs=output_audio)
        
        # 2. 流式开始
        start_evt = start_btn.click(
            fn=start_action, 
            inputs=[input_text, state],
            outputs=[data_transfer, state, start_btn, pause_btn],
            concurrency_limit = 4
        )
        
        # 3. 终止流式
        pause_btn.click(
            fn=pause_action, 
            inputs=[state],
            outputs=[data_transfer, state, start_btn, pause_btn],
            cancels=[start_evt]
        )
        
        # 4. JS 数据监听 (通过 data_transfer 组件的内容变化触发前端 js)
        data_transfer.change(fn=lambda x: x, inputs=data_transfer, outputs=None, js=js_code)
        
        # 隐藏组件与回调
        hidden_finish_btn = gr.Button("FINISH", elem_id="hidden_finish_btn")
        hidden_finish_btn.click(
            fn=finalize_ui,
            inputs=None,
            outputs=[start_btn, pause_btn]
        )

        # 页面 Footer HTML
        gr.HTML("""
            <hr style="border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0)); margin: 20px 0;">
            <div style="text-align: center; padding: 20px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">
                <p style="color: #555; font-weight: 500;">
                    👤 <b>Author:</b>
                    <a href="https://github.com/pengyichen2026" target="_blank" style="text-decoration: none; color: #ff9800; border-bottom: 1px dashed #ff9800;">彭亦宸</a>
                    &nbsp;·&nbsp;
                    📧 <b>Contact:</b>
                    <a href="mailto:pengyichenthu@gmail.com" style="display: inline-flex; align-items: center; background-color: #f1f1f1; padding: 4px 12px; border-radius: 20px; text-decoration: none; color: #d44638; font-weight: bold; transition: 0.3s;">
                        pengyichenthu@gmail.com
                    </a>
                </p>
            </div>
        """)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=6006)