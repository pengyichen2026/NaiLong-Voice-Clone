import os 
import torch 
import torchaudio 
import shutil 
import numpy as np 
from modelscope.pipelines import pipeline 
from modelscope.utils.constant import Tasks 
from tqdm import tqdm 

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

device = "cuda" if torch.cuda.is_available() else "cpu" 

print("正在初始化声纹模型...") 
sv_pipeline = pipeline(task=Tasks.speaker_verification,  
                       model='damo/speech_campplus_sv_zh-cn_16k-common', 
                       model_revision='v1.0.0', 
                       device=device) 

model = sv_pipeline.model 
model.to(device) 
model.eval() 

def get_embedding_direct(file_path): 
    wav, fs = torchaudio.load(file_path) 
    if fs != 16000: 
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000) 
        wav = resampler(wav) 
    wav = wav.mean(dim=0, keepdim=True).to(device) 
    with torch.no_grad(): 
        emb = model(wav) 
    return emb.squeeze().cpu().float() 

def save_and_trim(src, dest): 
    try:
        abs_src = os.path.abspath(src)
        abs_pool = os.path.abspath(POOL_DIR)
        rel_path = os.path.relpath(abs_src, abs_pool)
        print(f"正在处理相对路径: {rel_path}")
        safe_name = rel_path.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
        info = torchaudio.info(src) 
        duration = info.num_frames / info.sample_rate 
        if 1.2 <= duration <= 15.0: 
            out_path = os.path.join(dest, safe_name) 
            shutil.copy2(src, out_path) 
            return True 
    except Exception as e:
        print(f"保存文件 {src} 出错: {e}") # 调试时可以取消注释
        pass
    return False

# ====== 新增辅助函数：递归获取所有 wav 文件 ======
def get_all_audio_recursive(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(AUDIO_EXTENSIONS):
                # 把找到的 wav 文件拼接成完整路径
                audio_files.append(os.path.join(root, file))
    return audio_files
# ==================================================

def run_pipeline(): 
    print("注意：处理压缩格式 (.mp3/.m4a/.opus 等) 需预装 FFmpeg ！")
    os.makedirs(OUTPUT_DIR, exist_ok=True) 

    # 1. 提取黄金种子集 (修改为递归提取)
    print(f"正在读取黄金种子集: {SEED_DIR}") 
    seed_files = get_all_audio_recursive(SEED_DIR)

    seed_embs_list = [] 
    for f in tqdm(seed_files, desc="提取种子特征"): 
        try: 
            seed_embs_list.append(get_embedding_direct(f)) 
        except Exception as e: 
            print(f"\n文件 {f} 提取失败: {e}") 

    if not seed_embs_list: 
        print("错误：特征列表为空！") 
        return 

    # 每个种子权重为 1，所以初始总和就是向量求和
    seed_embs = torch.stack(seed_embs_list) # Shape: [N_seeds, 192]
    weighted_sum_emb = torch.sum(seed_embs, dim=0).to(device) 
    total_weight = float(len(seed_embs_list)) 
    
    # 初始重心 = 总和 / 总权重
    centroid = weighted_sum_emb / total_weight 
    print(f"种子提取完成！初始总权重: {total_weight}, 重心维度: {centroid.shape}") 

    # 2. 预提取池中特征 (修改为递归提取)
    print(f"正在预提取池中特征: {POOL_DIR}") 
    pool_files = get_all_audio_recursive(POOL_DIR)
    
    valid_files = [] 
    pool_embs_list = [] 
    
    for path in tqdm(pool_files, desc="池子特征化"): 
        try: 
            emb = get_embedding_direct(path)  
            valid_files.append(path) # 这里现在存的是完整绝对路径，而不仅仅是文件名
            pool_embs_list.append(emb) 
        except: 
            continue 
              
    all_embs_tensor = torch.stack(pool_embs_list).to(device) 
    active_mask = torch.ones(len(valid_files), dtype=torch.bool, device=device) 

    # 3. 迭代筛选 (每音频独立权重版)
    print("\n开始权重迭代筛选...") 
    for idx, t in enumerate(THRESHOLDS): 
        # 第 i 轮的每个音频权重为 0.8^(idx+1)
        current_unit_weight = DECAY_FACTOR ** (idx + 1)

        # 计算相似度
        scores = torch.nn.functional.cosine_similarity(centroid.unsqueeze(0), all_embs_tensor, dim=1) 
        
        # 找出符合条件的索引
        passed_indices = torch.where(active_mask & (scores >= t))[0] 
        added_count = 0 
        batch_sum_emb = torch.zeros_like(centroid)

        for i in passed_indices.tolist(): 
            src_path = valid_files[i] # 直接取出完整路径
            if save_and_trim(src_path, OUTPUT_DIR): 
                # 累加本轮符合条件的 embedding 向量
                batch_sum_emb += all_embs_tensor[i]
                added_count += 1 
            active_mask[i] = False 

        if added_count > 0: 
            # 更新全局加权总和：当前累加和 * 当前轮次的单位权重
            weighted_sum_emb += batch_sum_emb * current_unit_weight
            # 更新总权重分母：新增个数 * 单位权重
            total_weight += added_count * current_unit_weight 
            # 重新计算重心
            centroid = weighted_sum_emb / total_weight 
              
        print(f"第 {idx+1} 轮 [T={t:.2f}]: 新增 {added_count} 段 (该轮每段权重: {current_unit_weight:.4f}, 当前总权重: {total_weight:.2f})") 

    print(f"\n筛选完成！最终收敛重心基于总权重: {total_weight:.4f}") 

if __name__ == "__main__": 
    run_pipeline()