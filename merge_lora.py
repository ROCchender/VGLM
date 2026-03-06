#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
合并 LoRA 权重到基础模型
"""

import os
import sys
import argparse
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finetune_visualglm import FineTuneVisualGLMModel
from sat.training.model_io import save_checkpoint
from sat import get_args

# 配置路径
BASE_MODEL_PATH = "./root/.sat_models/visualglm-6b"  # 基础模型路径
LORA_MODEL_PATH = "./checkpoints/finetune-visualglm"  # LoRA 模型路径
OUTPUT_PATH = "./models/merged_model"  # 合并后模型保存路径


import json
import shutil


def merge_lora_to_base(base_model_path, model_path, output_path, model_name):

    print(f"\n{'='*60}")
    print(f"合并模型: {model_name}")
    print(f"{'='*60}")

    base_config_path = os.path.join(base_model_path, "config.json")
    if not os.path.exists(base_config_path):
        base_config_path = os.path.join(base_model_path, "model_config.json")
    
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            base_config = json.load(f)
        print(f"  读取基础模型配置: {base_config_path}")
    else:
        base_config = {}
        print(f"  警告: 未找到基础模型配置，使用默认值")

    source_config_path = os.path.join(model_path, "model_config.json")
    if os.path.exists(source_config_path):
        with open(source_config_path, 'r') as f:
            source_config = json.load(f)
        print(f"  读取 LoRA 模型配置: {source_config_path}")
    else:
        source_config = {}
        print(f"  警告: 未找到 LoRA 模型配置")
    
    # 步骤 1: 加载基础模型
    print(f"\n[1/4] 加载基础模型（VisualGLM-6B）...")

    lora_rank = source_config.get('lora_rank', 10)
    layer_range = source_config.get('layer_range', None)
    pre_seq_len = source_config.get('pre_seq_len', 4)
    num_layers = base_config.get('num_layers', 28)
    hidden_size = base_config.get('hidden_size', 4096)
    num_attention_heads = base_config.get('num_attention_heads', 32)
    
    print(f"  基础模型配置: num_layers={num_layers}, hidden_size={hidden_size}, num_attention_heads={num_attention_heads}")
    print(f"  LoRA 配置: lora_rank={lora_rank}, layer_range={layer_range}, pre_seq_len={pre_seq_len}")
    
    args = argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True if torch.cuda.is_available() else False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lora_rank=lora_rank,
        layer_range=layer_range,
        pre_seq_len=pre_seq_len,
        use_lora=True,
        use_qlora=False,
        use_ptuning=False,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
    )

    model, _ = FineTuneVisualGLMModel.from_pretrained(BASE_MODEL_PATH, args=args)
    print(f"✓ 基础模型加载完成")
    
    # 步骤 2: 加载 LoRA 权重
    print(f"\n[2/4] 加载 LoRA 权重从 {model_path}...")

    latest_file = os.path.join(model_path, "latest")
    if os.path.exists(latest_file):
        with open(latest_file, 'r') as f:
            iteration = f.read().strip()
        checkpoint_path = os.path.join(model_path, iteration, "mp_rank_00_model_states.pt")
    else:
        pt_files = []
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith('.pt'):
                    pt_files.append(os.path.join(root, file))
        if pt_files:
            checkpoint_path = pt_files[0]
        else:
            print(f"错误: 在 {model_path} 中找不到 checkpoint 文件")
            return None
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到 checkpoint 文件 {checkpoint_path}")
        return None
    
    print(f"  加载 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'module' in checkpoint:
        state_dict = checkpoint['module']
    else:
        state_dict = checkpoint

    lora_state_dict = {}
    for key, value in state_dict.items():
        if 'lora' in key.lower() or 'matrix_A' in key or 'matrix_B' in key:
            lora_state_dict[key] = value
    
    print(f"  找到 {len(lora_state_dict)} 个 LoRA 参数")

    model.load_state_dict(lora_state_dict, strict=False)
    print(f"✓ LoRA 权重加载完成")
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 步骤 3: 合并 LoRA 权重到基础模型
    print(f"\n[3/4] 合并 LoRA 权重到基础模型...")
    model.get_mixin('lora').merge_lora()
    print("✓ LoRA 合并完成")
    
    # 步骤 4: 保存合并后的完整模型
    print(f"\n[4/4] 保存合并后的模型到 {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    args.layer_range = []
    args.save = output_path
    args.mode = 'inference'
    
    save_checkpoint(1, model, None, None, args)
    print(f"✓ 模型权重保存完成")

    output_config_path = os.path.join(output_path, "model_config.json")
    merged_config = source_config.copy()
    merged_config['use_lora'] = False
    merged_config['use_qlora'] = False
    if 'layer_range' in merged_config:
        del merged_config['layer_range']
    with open(output_config_path, 'w') as f:
        json.dump(merged_config, f, indent=4)
    print(f"✓ 配置文件保存完成: {output_config_path}")
    
    # 清理内存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output_path


def main():
    print("=" * 60)
    print("VisualGLM LoRA 权重合并工具")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description='将 LoRA 权重合并到基础模型')
    parser.add_argument('--base_model', type=str, default=BASE_MODEL_PATH,
                        help='基础模型路径（默认使用 VisualGLM-6B）')
    parser.add_argument('--lora_model', type=str, default=LORA_MODEL_PATH,
                        help='LoRA 模型路径')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH,
                        help='合并后模型输出路径')
    parser.add_argument('--model_name', type=str, default='LoRA模型',
                        help='模型名称（用于显示）')
    
    args = parser.parse_args()

    if not os.path.exists(args.base_model):
        print(f"错误: 找不到基础模型路径 {args.base_model}")
        print("请确保基础模型已下载，或使用 --base_model 指定正确路径")
        return

    if not os.path.exists(args.lora_model):
        print(f"错误: 找不到 LoRA 模型路径 {args.lora_model}")
        return

    output = merge_lora_to_base(args.base_model, args.lora_model, args.output, args.model_name)
    if output:
        print(f"\n✓ 合并完成，完整模型保存到: {output}")
    else:
        print("\n✗ 合并失败")
    
    print("\n" + "=" * 60)
    print("合并完成！可以使用合并后的模型进行推理了。")
    print("=" * 60)


if __name__ == "__main__":
    main()
