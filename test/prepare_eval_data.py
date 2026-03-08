#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import json
import argparse
import os

def prepare_eval_data(coco_annotation_file: str, image_root: str, output_gt: str, 
                      num_samples: int = None):
    print(f"读取 COCO 标注: {coco_annotation_file}")
    with open(coco_annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # COCO 官方格式: images 和 annotations 分开
    images_info = {img['id']: img for img in data['images']}
    annotations = data['annotations']
    
    print(f"总图片数: {len(images_info)}")
    print(f"总标注数: {len(annotations)}")
    
    # 按图片 ID 分组 caption
    img_to_captions = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_to_captions:
            img_to_captions[img_id] = []
        img_to_captions[img_id].append(ann['caption'])
    
    # 限制样本数
    img_ids = list(img_to_captions.keys())
    if num_samples and num_samples < len(img_ids):
        img_ids = img_ids[:num_samples]
        print(f"采样数量: {num_samples}")
    
    ground_truths = []
    image_paths = []
    skipped = 0
    
    for i, img_id in enumerate(img_ids):
        new_img_id = f"img_{i:05d}"
        img_info = images_info[img_id]
        
        # 构建图片文件名 (COCO 格式: COCO_val2014_000000xxxxxx.jpg)
        filename = img_info['file_name']
        filepath = os.path.join(image_root, filename)
        
        # 检查图片是否存在
        if not os.path.exists(filepath):
            print(f"  ⚠️ 图片不存在，跳过: {filepath}")
            skipped += 1
            continue
        
        # 获取该图片的所有 caption (通常是5条)，并清洗特殊字符
        captions = [c.replace('\n', ' ').replace('\r', ' ').strip() for c in img_to_captions[img_id]]
        
        # 为每条 caption 创建 ground truth
        for caption in captions:
            ground_truths.append({
                "image_id": new_img_id,
                "caption": caption
            })
        
        # 图片路径只存一次
        image_paths.append({
            "image_id": new_img_id,
            "img_path": filepath,
            "prompt": "Describe this image in English."
        })
    
    if skipped > 0:
        print(f"\n  ⚠️ 共跳过 {skipped} 张不存在的图片")
    
    # 保存 ground truth
    with open(output_gt, 'w', encoding='utf-8') as f:
        json.dump(ground_truths, f, ensure_ascii=False, indent=2)
    
    # 保存图片列表
    output_img_list = output_gt.replace('.json', '_images.json')
    with open(output_img_list, 'w', encoding='utf-8') as f:
        json.dump(image_paths, f, ensure_ascii=False, indent=2)
    
    avg_captions = len(ground_truths) // len(image_paths) if image_paths else 0
    print(f"\n已生成评估数据:")
    print(f"  图片数量: {len(image_paths)} 张")
    print(f"  标注总数: {len(ground_truths)} 条 (平均每张图 {avg_captions} 条)")
    print(f"  标签文件: {output_gt}")
    print(f"  图片列表: {output_img_list}")
    print(f"\n下一步:")
    print(f"  1. 运行模型生成预测: python generate_predictions.py --img-list {output_img_list}")
    print(f"  2. 运行评估: python evaluate_model.py --pred predictions.json --gt {output_gt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='准备评估数据 (Ground Truth)')
    parser.add_argument('--coco-annotation', type=str, 
                        default='/gemini/data-1/caption_data/annotations/captions_val2014.json',
                        help='COCO 官方标注文件 (如 captions_val2014.json)')
    parser.add_argument('--image-root', type=str, 
                        default='/gemini/data-1/caption_data/val2014',
                        help='图片目录 (如 val2014/)')
    parser.add_argument('--output-gt', type=str, 
                        default='./eval_ground_truth.json',
                        help='输出标签文件')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='样本数量 (默认使用全部)')
    
    args = parser.parse_args()
    
    prepare_eval_data(args.coco_annotation, args.image_root, 
                      args.output_gt, args.num_samples)
