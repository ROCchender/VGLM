#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
合并 LoRA 权重
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finetune_visualglm import FineTuneVisualGLMModel
from sat.training.model_io import save_checkpoint

# 配置路径（请根据你的实际情况修改以下路径）
BASE_MODEL_PATH = "./visualglm"  # 基础模型路径
MODEL_PATH_1 = "./models/300VGLM"  # 第一个需要合并的模型路径（多模型合并时使用）
MODEL_PATH_2 = "./models/5000VGLM"  # 第二个需要合并的模型路径
OUTPUT_PATH = "./models/merged_300_5000"  # 合并后模型保存路径


import json
import shutil

def merge_single_lora(model_path, output_path, model_name):
    """合并单个 LoRA 模型"""
    print(f"\n{'='*60}")
    print(f"合并模型: {model_name}")
    print(f"{'='*60}")

    source_config_path = os.path.join(model_path, "model_config.json")
    if os.path.exists(source_config_path):
        with open(source_config_path, 'r') as f:
            source_config = json.load(f)
        print(f"  读取源模型配置: {source_config_path}")
    else:
        source_config = {}
        print(f"  警告: 未找到源模型配置")
    
    print(f"[1/3] 从 {model_path} 加载模型...")
    args = argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True if torch.cuda.is_available() else False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lora_rank=10,
        layer_range=[],
        pre_seq_len=4,
        use_lora=True,
        use_qlora=False,
    )
    
    model, _ = FineTuneVisualGLMModel.from_pretrained(model_path, args=args)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("✓ 模型加载完成")
    
    print(f"[2/3] 合并 LoRA 权重...")
    model.get_mixin('lora').merge_lora()
    print("✓ LoRA 合并完成")
    
    print(f"[3/3] 保存合并后的模型到 {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    args.layer_range = []
    args.save = output_path
    args.mode = 'inference'
    
    save_checkpoint(1, model, None, None, args)
    print(f"✓ 模型权重保存完成")

    output_config_path = os.path.join(output_path, "model_config.json")
    with open(output_config_path, 'w') as f:
        json.dump(source_config, f, indent=4)
    print(f"✓ 配置文件保存完成: {output_config_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output_path


def merge_multiple_lora(model_path_1, model_path_2, output_path):
    """合并多个 LoRA 模型
    
    Args:
        model_path_1: 第一个模型路径
        model_path_2: 第二个模型路径
        output_path: 合并后模型保存路径
    """
    print(f"\n{'='*60}")
    print("多 LoRA 权重合并")
    print(f"{'='*60}")

    if not os.path.exists(model_path_1):
        print(f"错误: 找不到模型路径 {model_path_1}")
        return None
    
    if not os.path.exists(model_path_2):
        print(f"错误: 找不到模型路径 {model_path_2}")
        return None

    source_config_path = os.path.join(model_path_1, "model_config.json")
    if os.path.exists(source_config_path):
        with open(source_config_path, 'r') as f:
            source_config = json.load(f)
        print(f"  读取源模型配置: {source_config_path}")
    else:
        source_config = {}
        print(f"  警告: 未找到源模型配置")
    
    model_name_1 = os.path.basename(model_path_1)
    model_name_2 = os.path.basename(model_path_2)
    print(f"[1/4] 加载基础模型（带 {model_name_1} LoRA）...")
    args = argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True if torch.cuda.is_available() else False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lora_rank=10,
        layer_range=[],
        pre_seq_len=4,
        use_lora=True,
        use_qlora=False,
    )

    model, _ = FineTuneVisualGLMModel.from_pretrained(model_path_1, args=args)
    
    if torch.cuda.is_available():
        model = model.cuda()

    print(f"✓ {model_name_1} 加载完成")

    print(f"[2/4] 临时加载 {model_name_2} 获取权重...")
    model_2, _ = FineTuneVisualGLMModel.from_pretrained(model_path_2, args=args)
    
    if torch.cuda.is_available():
        model_2 = model_2.cuda()
    
    print(f"✓ {model_name_2} 加载完成")
    
    print(f"[3/4] 融合两个 LoRA 权重（平均叠加）...")
    with torch.no_grad():
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(),
            model_2.named_parameters()
        ):
            if 'lora' in name1 and 'matrix' in name1:
                # 平均融合两个 LoRA
                param1.data = (param1.data + param2.data) / 2
                print(f"  融合: {name1}")
    
    print("✓ 权重融合完成")

    del model_2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"[4/4] 合并 LoRA 到基础模型...")
    model.get_mixin('lora').merge_lora()
    print("✓ LoRA 合并完成")
    
    print(f"[5/5] 保存合并后的模型到 {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    args.layer_range = []
    args.save = output_path
    args.mode = 'inference'
    
    save_checkpoint(1, model, None, None, args)
    print(f"✓ 模型权重保存完成")

    output_config_path = os.path.join(output_path, "model_config.json")
    with open(output_config_path, 'w') as f:
        json.dump(source_config, f, indent=4)
    print(f"✓ 配置文件保存完成: {output_config_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output_path


def main():
    print("=" * 60)
    print("VisualGLM LoRA 权重合并工具")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description='合并 VisualGLM LoRA 权重')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'multi'],
                        help='合并模式: single=合并单个模型, multi=合并两个模型')
    parser.add_argument('--model', type=str, default=None,
                        help='单个模型合并时的模型路径')
    parser.add_argument('--model1', type=str, default=None,
                        help='多模型合并时的第一个模型路径')
    parser.add_argument('--model2', type=str, default=None,
                        help='多模型合并时的第二个模型路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出路径（可选）')

    args = parser.parse_args()

    if args.mode == 'single':
        model_path = args.model or MODEL_PATH_2
        model_name = os.path.basename(model_path)
        output = args.output or f"./models/{model_name}_merged"
        merge_single_lora(model_path, output, model_name)
        print(f"\n✓ {model_name} 合并完成，保存到: {output}")

    elif args.mode == 'multi':
        model1_path = args.model1 or MODEL_PATH_1
        model2_path = args.model2 or MODEL_PATH_2
        output = args.output or OUTPUT_PATH
        output = merge_multiple_lora(model1_path, model2_path, output)
        if output:
            print(f"\n✓ 多模型合并完成，保存到: {output}")
        else:
            print("\n✗ 合并失败")
    
    print("\n" + "=" * 60)
    print("合并完成！可以使用合并后的模型进行推理了。")
    print("=" * 60)


if __name__ == "__main__":
    main()
