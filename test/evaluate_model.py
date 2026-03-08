#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import json
import argparse
import re
import os
import tempfile
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def main():
    parser = argparse.ArgumentParser(description='使用官方 COCO 工具评估模型')
    parser.add_argument('--pred', type=str, required=True,
                        help='预测结果文件, e.g. val/eval_predictions30000.json')
    parser.add_argument('--gt', type=str, required=True,
                        help='官方 COCO 标注文件, e.g. caption_data/captions_val2014.json')
    parser.add_argument('--mapping', type=str, default=None,
                        help='img_id 到图片路径的映射文件, e.g. val/eval_ground_truth_images.json')

    args = parser.parse_args()

    print("=" * 60)
    print("图像描述模型评估 - 官方 COCO 标准评测")
    print("=" * 60)

    print(f"\n加载官方标注: {args.gt}")
    coco = COCO(args.gt)
    print(f"  GT 图片数: {len(coco.imgs)}")
    print(f"  GT 标注数: {len(coco.anns)}")

    print(f"\n加载预测结果: {args.pred}")
    with open(args.pred, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    print(f"  预测样本数: {len(preds)}")

    if args.mapping is None:
        pred_dir = os.path.dirname(os.path.abspath(args.pred))
        args.mapping = os.path.join(pred_dir, 'eval_ground_truth_images.json')

    print(f"\n加载映射文件: {args.mapping}")
    with open(args.mapping, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    imgstr_to_cocoid = {}
    for item in mapping_data:
        img_id_str = item['image_id']       
        img_path = item['img_path']         
        filename = os.path.basename(img_path)

        numbers = re.findall(r'\d+', filename)
        if numbers:
            coco_num_id = int(numbers[-1])
        else:
            print(f"  ⚠️ 无法从 {filename} 提取数字 ID，跳过")
            continue

        imgstr_to_cocoid[img_id_str] = coco_num_id

    print(f"  映射条目数: {len(imgstr_to_cocoid)}")

    pred_list = []
    skipped = 0
    for p in preds:
        img_id_str = p['image_id']
        if img_id_str in imgstr_to_cocoid:
            coco_id = imgstr_to_cocoid[img_id_str]
            if coco_id in coco.imgs:
                pred_list.append({
                    "image_id": coco_id,
                    "caption": p['caption']
                })
            else:
                skipped += 1
        else:
            skipped += 1

    if skipped > 0:
        print(f"  ⚠️ 有 {skipped} 条预测未在 GT 中找到对应图片，已跳过")
    print(f"  参与评测的预测数: {len(pred_list)}")

    if len(pred_list) == 0:
        print("\n❌ 没有可评测的样本！请确认映射文件和 GT 文件的图片 ID 一致。")
        return

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f_pred:
        json.dump(pred_list, f_pred, ensure_ascii=False)
        tmp_pred_path = f_pred.name

    try:
        with open(tmp_pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
            
        coco_results = coco.loadRes(pred_data)

        coco_eval = COCOEvalCap(coco, coco_results)
        coco_eval.params['image_id'] = coco_results.getImgIds()

        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Cider(), "CIDEr"),
        ]

        imgIds = coco_eval.params['image_id']
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = coco.imgToAnns[imgId]
            res[imgId] = coco_results.imgToAnns[imgId]

        print("  正在对文本进行基础分词预处理...")
        import string
        def simple_tokenize(data_dict):
            translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            tokenized_dict = {}
            for img_id, anns in data_dict.items():
                tokenized_anns = []
                for ann in anns:
                    txt = str(ann.get('caption', ''))
                    txt = txt.lower().translate(translator)
                    txt = ' '.join(txt.split())
                    tokenized_anns.append(txt)
                tokenized_dict[img_id] = tokenized_anns
            return tokenized_dict

        gts = simple_tokenize(gts)
        res = simple_tokenize(res)

        results = {}
        for scorer, method in scorers:
            print(f"  计算 {method}...")
            try:
                score, scores = scorer.compute_score(gts, res)
                if isinstance(method, list):
                    for sc, m in zip(score, method):
                        results[m] = sc
                else:
                    results[method] = score
            except Exception as e:
                print(f"    ⚠️ {method} 计算失败: {e}")

        print("\n" + "=" * 60)
        print("🎯 最终评估结果 (毕业设计核心指标):")
        print("=" * 60)

        for metric in ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "CIDEr"]:
            if metric in results:
                print(f"  {metric:>10s} : {results[metric]:.4f}")
        print("=" * 60)

        print("\n指标说明 (用于撰写论文论述):")
        print("  Bleu_4: 评估生成的文本与真实标注在 4-gram 层面的精确匹配程度，常用于衡量文本的连贯性。")
        print("  CIDEr : 专为图像描述设计的共识指标，通过 TF-IDF 赋予重要词汇更高权重，是目前最受认可的核心评价标准。")

    except Exception as e:
        print(f"\n❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if os.path.exists(tmp_pred_path):
            os.remove(tmp_pred_path)


if __name__ == '__main__':
    main()
