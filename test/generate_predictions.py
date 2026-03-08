#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import json
import argparse
import torch
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_predictions(model_path: str, img_list_file: str, output_file: str, 
                         max_length: int = 256, temperature: float = 0.8):

    print("="*60)
    print("生成预测结果")
    print("="*60)

    print(f"\n加载图片列表: {img_list_file}")
    with open(img_list_file, 'r', encoding='utf-8') as f:
        img_list = json.load(f)
    print(f"  共 {len(img_list)} 张图片")

    print(f"\n加载模型: {model_path}")
    
    # 这里需要根据你的实际模型加载方式修改
    # 示例使用 SAT 框架加载
    try:
        from finetune_visualglm import FineTuneVisualGLMModel
        from sat.model import AutoModel
        from sat.model.mixins import CachedAutoregressiveMixin
        from model import chat
        from transformers import AutoTokenizer

        tokenizer_path = "./visualglm"
        if not os.path.exists(tokenizer_path):
            tokenizer_path = "THUDM/visualglm-6b"
        
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print(f"  Tokenizer 加载完成")

        model, model_args = AutoModel.from_pretrained(
            model_path,
            args=argparse.Namespace(
                fp16=True,
                skip_init=True,
                use_gpu_initialization=torch.cuda.is_available(),
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
        )
        model = model.eval()
        model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        print(f"  模型加载完成 (设备: {'CUDA' if torch.cuda.is_available() else 'CPU'})")
        
    except Exception as e:
        print(f"  模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        print("  请根据实际模型类型修改加载代码")
        return

    print(f"\n开始生成预测...")
    predictions = []
    
    for item in tqdm(img_list, desc="生成进度"):
        img_id = item['image_id']
        img_path = item['img_path']
        prompt = item['prompt']
        
        try:
            response, _, _ = chat(
                image_path=img_path,
                model=model,
                tokenizer=tokenizer,
                query=prompt,
                history=[],
                max_length=max_length,
                temperature=temperature,
                top_p=0.7,
                top_k=30,
                english=True
            )
            
            predictions.append({
                "image_id": img_id,
                "caption": response.strip()
            })
            
        except Exception as e:
            print(f"\n  警告: 图片 {img_id} 生成失败: {e}")
            predictions.append({
                "image_id": img_id,
                "caption": ""
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f"\n预测结果已保存: {output_file} ({len(predictions)} 条)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='使用模型生成预测结果')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型路径 (如 ./checkpoints/finetune-visualglm-6b-xxx/11300)')
    parser.add_argument('--img-list', type=str, required=True,
                        help='图片列表文件 (由 prepare_eval_data.py 生成)')
    parser.add_argument('--output', type=str, default='./eval_predictions.json',
                        help='输出预测文件')
    parser.add_argument('--max-length', type=int, default=256,
                        help='生成文本最大长度')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='采样温度')
    
    args = parser.parse_args()
    
    generate_predictions(
        model_path=args.model_path,
        img_list_file=args.img_list,
        output_file=args.output,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    print("\n使用说明:")
    print("  1. 确保模型路径正确 (合并后的模型或 checkpoint)")
    print("  2. 确保 img-list 文件由 prepare_eval_data.py 生成")
    print("  3. 运行评估: python evaluate_model.py --pred eval_predictions.json --gt eval_ground_truth.json")


if __name__ == '__main__':
    main()
