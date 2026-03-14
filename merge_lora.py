#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
合并 LoRA 权重到基础模型（sat 框架模型）
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finetune_visualglm import FineTuneVisualGLMModel
from sat.training.model_io import save_checkpoint

# 默认配置路径
DEFAULT_LORA_MODEL_PATH = "/gemini/pretrain/"  # 请替换为实际你跑完的路径
DEFAULT_OUTPUT_PATH = "./models/merge_model"

def main():
    print("=" * 60)
    print("sat框架模型 LoRA 权重合并工具")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description='将 LoRA 胖检查点直接合并为标准模型')
    parser.add_argument('--lora_model', type=str, default=DEFAULT_LORA_MODEL_PATH,
                        help='训练跑完出来的模型路径')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_PATH,
                        help='合并后最终模型将要被保存的路径')
    
    args_cli = parser.parse_args()

    if not os.path.exists(args_cli.lora_model):
        print(f"错误: 找不到训练输出文件夹 {args_cli.lora_model}")
        print("请检查路径是否正确，或者是否已经完成了训练。")
        return

    print(f"\n[1/3] 从 {args_cli.lora_model} 加载携带 LoRA 结构的检查点...")

    args = argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True if torch.cuda.is_available() else False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    model, args = FineTuneVisualGLMModel.from_pretrained(args_cli.lora_model, args=args)
    print(f"✓ 包含 LoRA 的本体模型加载完成")
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    print(f"\n[2/3] 执行数学合并")
    model.get_mixin('lora').merge_lora()
    print("✓ 数学矩阵合并完成")
    
    print(f"\n[3/3] 覆盖清理配置，并保存为纯净版完整模型到 {args_cli.output}...")
    os.makedirs(args_cli.output, exist_ok=True)
    
    args.layer_range = []
    args.save = args_cli.output
    args.mode = 'inference'

    save_checkpoint(1, model, None, None, args)
    print(f"\n✓ 合并完成，纯净模型已保存至: {args_cli.output}")


if __name__ == "__main__":
    main()
