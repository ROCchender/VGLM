"""
COCO数据集转换脚本
将原版 COCO 标注 (captions_train2014.json / captions_val2014.json)转换为 VisualGLM 微调所需的 JSON 格式

用法:
    python prepare_coco_dataset.py \
        --annotation_dir /path/to/COCO2014/annotations \
        --image_root /path/to/COCO2014 \
        --output_dir /gemini/code/VGLM/coco_finetune_v2 \
        --max_samples 5000
"""

import json
import os
import argparse
import random
from pathlib import Path
from collections import defaultdict


def convert_coco_to_visualglm(annotation_dir, image_root, output_dir,
                               max_samples=None, seed=42, prompt="Describe this image in English."):
    random.seed(seed)

    train_ann_path = os.path.join(annotation_dir, "captions_train2014.json")
    val_ann_path = os.path.join(annotation_dir, "captions_val2014.json")

    # ---------- 读取训练集标注 ----------
    print(f"读取训练集标注: {train_ann_path}")
    with open(train_ann_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # 按 image_id 聚合所有 caption
    train_captions = defaultdict(list)
    for ann in train_data['annotations']:
        train_captions[ann['image_id']].append(ann['caption'].strip())

    # 构建 image_id -> file_name 映射
    train_id2file = {img['id']: img['file_name'] for img in train_data['images']}
    print(f"  训练集图片数: {len(train_id2file)}")
    print(f"  训练集标注数: {len(train_data['annotations'])}")

    # ---------- 读取验证集标注 ----------
    print(f"读取验证集标注: {val_ann_path}")
    with open(val_ann_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    val_captions = defaultdict(list)
    for ann in val_data['annotations']:
        val_captions[ann['image_id']].append(ann['caption'].strip())

    val_id2file = {img['id']: img['file_name'] for img in val_data['images']}
    print(f"  验证集图片数: {len(val_id2file)}")
    print(f"  验证集标注数: {len(val_data['annotations'])}")

    # ---------- 构建训练项 ----------
    train_items = []
    for image_id, file_name in train_id2file.items():
        captions = train_captions.get(image_id, [])
        if not captions:
            continue
        caption = random.choice(captions)
        filepath = os.path.join(image_root, "train2014", file_name)
        train_items.append({
            "img": filepath,
            "prompt": prompt,
            "label": caption
        })

    # ---------- 构建验证项 ----------
    val_items = []
    for image_id, file_name in val_id2file.items():
        captions = val_captions.get(image_id, [])
        if not captions:
            continue
        caption = random.choice(captions)
        filepath = os.path.join(image_root, "val2014", file_name)
        val_items.append({
            "img": filepath,
            "prompt": prompt,
            "label": caption
        })

    print(f"\n训练集样本: {len(train_items)}")
    print(f"验证集样本: {len(val_items)}")

    # ---------- 采样限制 ----------
    if max_samples and max_samples < len(train_items):
        random.shuffle(train_items)
        train_items = train_items[:max_samples]
        print(f"已采样训练集: {max_samples} 条")

    if max_samples and max_samples < len(val_items):
        val_samples = min(max_samples // 10, len(val_items))
        random.shuffle(val_items)
        val_items = val_items[:val_samples]
        print(f"已采样验证集: {val_samples} 条")

    # ---------- 验证图片是否存在（检查前5张） ----------
    print("\n验证图片路径...")
    check_count = min(5, len(train_items))
    for i in range(check_count):
        path = train_items[i]['img']
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {path}")
        if not exists and i == 0:
            print(f"  警告: 图片不存在，请检查 --image_root 路径是否正确")

    # ---------- 混入身份认知训练数据 ----------
    identity_qa = [
        {"prompt": "你是谁？", "label": "我是VGLM，一个轻量化图像描述模型，可以帮助你理解和描述图片内容。"},
        {"prompt": "你叫什么名字？", "label": "我叫VGLM，是一个专注于图像描述的轻量化多模态模型。"},
        {"prompt": "介绍一下你自己。", "label": "我是VGLM，一个轻量化的视觉语言模型。我能够理解图片内容并用自然语言进行描述，支持中文和英文的图像问答。"},
        {"prompt": "你是什么模型？", "label": "我是VGLM，一个轻量化多模态对话模型，能够理解图像并进行中英文对话。"},
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
    print(f"已混入身份认知数据: {len(identity_items)} 条 (模型将学会自称 VGLM)")

    # ---------- 保存 ----------
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
    parser.add_argument('--annotation_dir', type=str,
                        default='/path/to/COCO2014/annotations',
                        help='COCO annotations 目录 (包含 captions_train2014.json 和 captions_val2014.json)')
    parser.add_argument('--image_root', type=str,
                        default='/path/to/COCO2014',
                        help='COCO 图片根目录 (包含 train2014/ 和 val2014/ 子目录)')
    parser.add_argument('--output_dir', type=str,
                        default='/gemini/code/VGLM/coco_finetune',
                        help='输出目录')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大训练样本数 (用于小规模测试)')
    parser.add_argument('--prompt', type=str, default='Describe this image in English.',
                        help='统一的提示词')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    convert_coco_to_visualglm(
        annotation_dir=args.annotation_dir,
        image_root=args.image_root,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        prompt=args.prompt
    )
