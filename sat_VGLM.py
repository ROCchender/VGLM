#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys
import platform
import signal
import gc
import torch
import argparse
import requests
import tempfile

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

gc.collect()

# 默认模型路径（请修改为你自己的模型路径）
DEFAULT_MODEL_PATH = "./models/5000VGLM_merged"


def print_header():
    """打印标题"""
    print("=" * 60)
    print("VGLM - 图像描述模型 (SAT框架)")
    print("=" * 60)


def load_model(model_path, use_quant=False, quant_bits=4):
    """加载模型 (SAT方式)
    
    Args:
        model_path: 模型路径
        use_quant: 是否使用量化
        quant_bits: 量化位数 (4 或 8)
    """
    print(f"\n[1/2] 加载 Tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "./visualglm", 
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✓ Tokenizer 加载完成")
    
    print(f"\n[2/2] 加载模型 (SAT)...")
    if use_quant:
        print(f"  使用 {quant_bits}-bit 量化")
    
    from sat.model import AutoModel
    from sat.model.mixins import CachedAutoregressiveMixin
    from sat.quantization.kernels import quantize

    from finetune_visualglm import FineTuneVisualGLMModel

    model, model_args = AutoModel.from_pretrained(
        model_path,
        args=argparse.Namespace(
            fp16=True,
            skip_init=True,
            use_gpu_initialization=True if (torch.cuda.is_available() and not use_quant) else False,
            device='cuda' if (torch.cuda.is_available() and not use_quant) else 'cpu',
        )
    )
    model = model.eval()
    
    # 量化
    if use_quant and quant_bits in [4, 8]:
        quantize(model, quant_bits)
        if torch.cuda.is_available():
            model = model.cuda()
    
    # 添加自动回归 mixin
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    
    print("✓ 模型加载完成")
    
    if torch.cuda.is_available():
        print(f"\nGPU 内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return model, tokenizer


def download_image(url):
    """下载在线图片到临时文件"""
    try:
        print(f"  正在下载图片: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '')
        if 'jpeg' in content_type or 'jpg' in content_type:
            ext = '.jpg'
        elif 'png' in content_type:
            ext = '.png'
        elif 'gif' in content_type:
            ext = '.gif'
        elif 'webp' in content_type:
            ext = '.webp'
        else:
            ext = '.jpg'

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_file.write(response.content)
        temp_file.close()
        
        print(f"  ✓ 图片下载完成")
        return temp_file.name
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return None


def is_image_input(text):
    """检测输入是否为图片路径或图片URL"""
    text = text.strip()
    if text.startswith(('http://', 'https://')):
        return True
    if any(text.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif')):
        return True
    if os.path.isfile(text):
        return True
    return False


def safe_input(prompt_text):
    """安全的输入函数，处理编码问题"""
    try:
        import sys
        sys.stdout.buffer.write(prompt_text.encode('utf-8'))
        sys.stdout.flush()
        user_input = input()
        return user_input
    except UnicodeDecodeError:
        try:
            user_input = input()
            return user_input.encode('latin-1').decode('utf-8')
        except:
            return user_input
    except Exception as e:
        print(f"输入错误: {e}")
        return ""


def chat_with_model(model, tokenizer, image_path, query, history, args):
    """与模型对话"""
    from model import chat
    
    response, history, cache_image = chat(
        image_path, 
        model, 
        tokenizer,
        query, 
        history=history, 
        image=None,
        max_length=args.max_length, 
        top_p=args.top_p, 
        temperature=args.temperature,
        top_k=args.top_k,
        english=args.english,
        invalid_slices=[slice(63823, 130000)] if args.english else []
    )

    if isinstance(response, bytes):
        response = response.decode('utf-8', errors='replace')
    elif isinstance(response, str):
        response = response.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    
    return response, history, cache_image


def main():
    parser = argparse.ArgumentParser(description='VGLM 图像描述模型 (SAT)')
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, 
                        help=f'模型路径 (默认: {DEFAULT_MODEL_PATH})')
    parser.add_argument("--max_length", type=int, default=2048, 
                        help='最大序列长度 (默认: 2048)')
    parser.add_argument("--top_p", type=float, default=0.4, 
                        help='top p 采样 (默认: 0.4)')
    parser.add_argument("--top_k", type=int, default=100, 
                        help='top k 采样 (默认: 100)')
    parser.add_argument("--temperature", type=float, default=0.8, 
                        help='温度 (默认: 0.8)')
    parser.add_argument("--english", action='store_true', 
                        help='英文输出模式')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, 
                        help='量化位数 (4 或 8，用于节省显存)')
    parser.add_argument("--prompt_zh", type=str, default="描述这张图片。", 
                        help='中文提示词')
    parser.add_argument("--prompt_en", type=str, default="Describe the image.", 
                        help='英文提示词')
    
    args = parser.parse_args()
    
    print_header()

    model, tokenizer = load_model(args.model_path, use_quant=args.quant is not None, quant_bits=args.quant)

    print("\n" + "=" * 60)
    if args.english:
        print("Ready!")
        print("Commands: clear = restart, stop = exit")
    else:
        print("准备就绪！")
        print("提示词建议：")
        print("  1. '详细描述这张图片的背景和环境' - 侧重背景")
        print("  2. '描述这张图片' - 通用描述")
        print("  3. 'Describe this image' - 英文描述")
        print("命令：clear 清空对话历史，stop 终止程序")
    print("=" * 60)

    stop_stream = False
    
    while True:
        history = None
        cache_image = None
        temp_image_path = None

        if args.english:
            image_input = safe_input("\nEnter image path or URL (press Enter for text only): ")
        else:
            image_input = safe_input("\n请输入图片路径或URL（回车进入纯文本对话）：")
        
        if image_input.lower() == 'stop':
            break

        if image_input.startswith(('http://', 'https://')):
            temp_image_path = download_image(image_input)
            if temp_image_path is None:
                continue
            image_path = temp_image_path
        elif image_input and os.path.isfile(image_input):
            image_path = image_input
        else:
            image_path = None

        if image_path:
            query = args.prompt_en if args.english else args.prompt_zh
        else:
            if args.english:
                query = safe_input("User: ")
            else:
                query = safe_input("用户：")

        while True:
            if query.lower() == 'clear':
                if temp_image_path and os.path.exists(temp_image_path):
                    try:
                        os.unlink(temp_image_path)
                    except:
                        pass
                break
            
            if query.lower() == 'stop':
                sys.exit(0)
            
            try:
                response, history, cache_image = chat_with_model(
                    model, tokenizer, image_path, query, history, args
                )
                
                sep = 'A:' if args.english else '答：'
                print(f"VGLM：{response.split(sep)[-1].strip()}")
                
            except Exception as e:
                print(f"Error: {e}")
                break

            image_path = None

            if args.english:
                query = safe_input("User: ")
            else:
                query = safe_input("用户：")

            if is_image_input(query):
                print("\n检测到新图片，正在切换...\n")

                if temp_image_path and os.path.exists(temp_image_path):
                    try:
                        os.unlink(temp_image_path)
                    except:
                        pass

                if query.startswith(('http://', 'https://')):
                    temp_image_path = download_image(query.strip())
                    if temp_image_path is None:
                        continue
                    image_path = temp_image_path
                else:
                    image_path = query.strip()

                history = None
                cache_image = None
                query = args.prompt_en if args.english else args.prompt_zh


if __name__ == "__main__":
    main()
