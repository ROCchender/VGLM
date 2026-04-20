"""
Flickr30k 数据集转换脚本
将 Flickr30k 的标注文件转换为 VisualGLM 微调所需的 JSON 格式

用法:
    python prepare_flickr30k_dataset.py \
        --token_path /path/to/flickr30k/results_20130124.token \
        --image_root /path/to/flickr30k-images \
        --output_dir ./flickr30k_finetune
"""

import json
import os
import argparse
import random
from collections import defaultdict

def convert_flickr_to_visualglm(token_path, image_root, output_dir, max_samples=None, seed=42, prompt="Describe this image:"):
    random.seed(seed)
    
    print(f"读取 Flickr30k 标注: {token_path}")
    if not os.path.exists(token_path):
        print(f"错误: 找不到标注文件 {token_path}")
        return
        
    with open(token_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    captions_dict = defaultdict(list)
    for line in lines:
        if '\t' not in line:
            continue
        # 格式如: 1000092795.jpg#0   Two young guys with shaggy hair look at their hands while hanging out in the yard .
        img_id_with_num, caption = line.strip().split('\t', 1)
        img_filename = img_id_with_num.split('#')[0]
        captions_dict[img_filename].append(caption)
        
    all_images = list(captions_dict.keys())
    random.shuffle(all_images)
    
    # 标准划分: 1000测试, 1000验证, 剩余训练 (约29000)
    val_size = 1000
    val_images = all_images[:val_size]
    train_images = all_images[val_size:]
    
    def build_items(images):
        items = []
        for img_name in images:
            filepath = os.path.join(image_root, img_name)
            # 每个图片可能有多条 caption，随机选一条或者全部展开（通常随机选一条作多 epoch，或者展开全部当大样本）
            # 这里统一采用随机选其中一条作为一次训练的 label
            caption = random.choice(captions_dict[img_name])
            items.append({
                "img": filepath,
                "prompt": prompt,
                "label": caption
            })
        return items
        
    train_items = build_items(train_images)
    val_items = build_items(val_images)
    
    print(f"\n训练集样本: {len(train_items)}")
    print(f"验证集样本: {len(val_items)}")
    
    if max_samples and max_samples < len(train_items):
        train_items = train_items[:max_samples]
        print(f"已采样训练集: {max_samples} 条")
        
    if max_samples and max_samples < len(val_items):
        val_samples = min(max_samples // 10, len(val_items))
        val_items = val_items[:val_samples]
        print(f"已采样验证集: {val_samples} 条")
        
    # 添加身份认知数据
    identity_qa = [
        {"prompt": "你是谁？", "label": "我是VGLM，一个轻量化图像描述模型，可以帮助你理解和描述图片内容。"},
        {"prompt": "你能做什么？", "label": "我是VGLM，我可以帮你描述图片内容、回答关于图片的问题，支持中文和英文的视觉问答。"},
    ]
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
    print(f"已混入身份认知数据: {len(identity_items)} 条")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "flickr30k_train.json")
    val_path = os.path.join(output_dir, "flickr30k_val.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_items, f, ensure_ascii=False, indent=2)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_items, f, ensure_ascii=False, indent=2)
        
    print(f"\n✓ 训练集已保存: {train_path}")
    print(f"✓ 验证集已保存: {val_path}")
    
    print("\n--- 训练数据样例 ---")
    for item in train_items[:3]:
        print(f"  图片: {os.path.basename(item['img'])}")
        print(f"  提示: {item['prompt']}")
        print(f"  标签: {item['label']}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_path', type=str, required=True, help='results_20130124.token 路径')
    parser.add_argument('--image_root', type=str, required=True, help='flickr30k-images 图片目录')
    parser.add_argument('--output_dir', type=str, default='./flickr30k_finetune', help='输出目录')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--prompt', type=str, default='Describe this image.')
    args = parser.parse_args()
    
    convert_flickr_to_visualglm(args.token_path, args.image_root, args.output_dir, args.max_samples, prompt=args.prompt)
