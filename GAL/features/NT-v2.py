import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import os

# ================= 1. 配置模型 (NT v2-100M) =================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

print(f"正在加载模型至 {device} (NT v2-100M)...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()

# ================= 2. 全量 Embedding 提取函数 =================
def extract_full_embeddings(sequences, batch_size=16):
    all_embeddings = []
    
    # 禁用梯度计算，加速推理并节省显存
    for i in tqdm(range(0, len(sequences), batch_size), desc="提取 512 维语义特征"):
        # 预处理序列
        batch_seqs = [s.upper().replace('U', 'T') for s in sequences[i : i + batch_size]]
        
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        # 提取最后一层隐藏状态 [Batch, Length, 512]
        last_hidden = outputs.hidden_states[-1]
        
        # 使用 Attention Mask 进行 Mean Pooling (排除 Padding 干扰)
        mask = inputs['attention_mask'].unsqueeze(-1)  # [Batch, Length, 1]
        sum_hidden = torch.sum(last_hidden * mask, dim=1)
        mean_hidden = sum_hidden / torch.sum(mask, dim=1)
        
        all_embeddings.append(mean_hidden.cpu().numpy())
        
    return np.vstack(all_embeddings)

# ================= 3. 主程序 =================
def main():
    # 输入你之前的物理特征表（包含 spacer 列）
    input_file = "GAL/data/spacer_data.csv"
    output_file = "spacer_NT_features.csv"
    
    if not os.path.exists(input_file):
        print(f"未找到输入文件 {input_file}")
        return

    df = pd.read_csv(input_file)
    sequences = df['spacer'].tolist()

    # 1. 提取全量 512 维 Embedding
    print(f"开始处理 {len(sequences)} 条数据...")
    raw_embeddings = extract_full_embeddings(sequences)
    
    # 2. 构造特征列名 (NT_v2_1 ... NT_v2_512)
    embedding_cols = [f'NT_v2_{i+1}' for i in range(raw_embeddings.shape[1])]
    embedding_df = pd.DataFrame(raw_embeddings, columns=embedding_cols)

    # 3. 合并数据：保留原始 spacer 和物理特征，并加入全量 LLM 特征
    final_df = pd.concat([df, embedding_df], axis=1)

    # 4. 保存
    final_df.to_csv(output_file, index=False)
    print(f"\n--- 处理完成 ---")
    print(f"全量特征已保存至: {output_file}")
    print(f"特征矩阵维度: {final_df.shape} (总计 {len(embedding_cols)} 个语义特征)")

if __name__ == "__main__":
    main()