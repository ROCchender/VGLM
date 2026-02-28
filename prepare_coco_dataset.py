"""
COCO数据集转换脚本（带身份认知训练，有需要可以自己去掉对应的注释）
将 Karpathy Split 格式的 dataset_coco.json 转换为 VisualGLM 微调所需的 JSON 格式

用法:
    python prepare_coco_dataset.py \
        --annotation /gemini/data-1/caption_data/dataset_coco.json \  # 请修改为你自己的标注文件路径
        --image_root /gemini/data-1/dataset \                         # 请修改为你自己的图片根目录路径
        --output_dir /gemini/data-1/coco_finetune \                   # 请修改为你自己的输出目录路径
        --max_samples 5000
"""

import json
import os
import argparse
import random
from pathlib import Path


def convert_karpathy_to_visualglm(annotation_path, image_root, output_dir, 
                                   max_samples=None, seed=42, prompt="描述这张图片。"):
    random.seed(seed)
    
    print(f"读取标注文件: {annotation_path}")
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images = data['images']
    print(f"总图片数: {len(images)}")

    train_items = []
    val_items = []
    
    for img in images:
        split = img['split']
        filepath = os.path.join(image_root, img['filepath'], img['filename'])

        if img['sentences']:
            caption = random.choice(img['sentences'])['raw'].strip()
        else:
            continue
        
        item = {
            "img": filepath,
            "prompt": prompt,
            "label": caption
        }
        
        if split in ('train', 'restval'):
            train_items.append(item)
        elif split == 'val':
            val_items.append(item)
    
    print(f"训练集图片: {len(train_items)}")
    print(f"验证集图片: {len(val_items)}")
    
    # 采样限制
    if max_samples and max_samples < len(train_items):
        random.shuffle(train_items)
        train_items = train_items[:max_samples]
        print(f"已采样训练集: {max_samples} 条")
    
    if max_samples and max_samples < len(val_items):
        val_val_samples = min(max_samples // 10, len(val_items))
        random.shuffle(val_items)
        val_items = val_items[:val_val_samples]
        print(f"已采样验证集: {val_val_samples} 条")
    
    # 验证图片是否存在（检查前5张）
    print("\n验证图片路径...")
    check_count = min(5, len(train_items))
    for i in range(check_count):
        path = train_items[i]['img']
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {path}")
        if not exists and i == 0:
            print(f"  警告: 图片不存在，请检查 --image_root 路径是否正确")
    
    # [可选] 混入身份认知训练数据，让模型学会自称
    # 如需启用，请取消以下代码块的注释
    """
    identity_qa = [
        {"prompt": "你是谁？", "label": "我是VGLM，一个轻量化图像描述模型，可以帮助你理解和描述图片内容。"},
        {"prompt": "你叫什么名字？", "label": "我叫VGLM，是一个专注于图像描述的轻量化多模态模型。"},
        {"prompt": "介绍一下你自己。", "label": "我是VGLM，一个轻量化的视觉语言模型。我能够理解图片内容并用自然语言进行描述，支持中文和英文的图像问答。"},
        {"prompt": "你是什么模型？", "label": "我是VGLM，一个轻量化多模态对话模型，能够理解图像并进行中英文对话。"},
        {"prompt": "你能做什么？", "label": "我是VGLM，我可以帮你描述图片内容、回答关于图片的问题，支持中文和英文的视觉问答。"},
    ]
    
    # 每条身份问答重复多次并绑定随机图片，确保模型充分学习
    identity_repeat = max(20, len(train_items) // 500)
    identity_items = []
    for _ in range(identity_repeat):
        for qa in identity_qa:
            random_img = random.choice(train_items)['img']
            identity_items.append({
                "img": random_img,
                "prompt": qa["prompt"],
                "label": qa["label"]
            })
    
    train_items.extend(identity_items)
    random.shuffle(train_items)
    print(f"已混入身份认知数据: {len(identity_items)} 条 (模型将学会自称 VGLM)")
    """

    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "coco_train.json")
    val_path = os.path.join(output_dir, "coco_val.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_items, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 训练集已保存: {train_path} ({len(train_items)} 条)")
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_items, f, ensure_ascii=False, indent=2)
    print(f"✓ 验证集已保存: {val_path} ({len(val_items)} 条)")

    print("\n--- 训练数据样例 ---")
    for item in train_items[:3]:
        print(f"  图片: {os.path.basename(item['img'])}")
        print(f"  提示: {item['prompt']}")
        print(f"  标签: {item['label']}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='转换 COCO 2014 数据集为 VisualGLM 微调格式')
    parser.add_argument('--annotation', type=str, 
                        default='/gemini/data-1/caption_data/dataset_coco.json',
                        help='Karpathy split JSON 路径')
    parser.add_argument('--image_root', type=str, 
                        default='/gemini/data-1/dataset',
                        help='COCO 图片根目录 (包含 train2014/ 和 val2014/)')
    parser.add_argument('--output_dir', type=str, 
                        default='/gemini/code/VGLM/coco_finetune',
                        help='输出目录')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大训练样本数 (用于小规模测试)')
    parser.add_argument('--prompt', type=str, default='描述这张图片。',
                        help='统一的提示词')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    convert_karpathy_to_visualglm(
        annotation_path=args.annotation,
        image_root=args.image_root,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        prompt=args.prompt
    )
