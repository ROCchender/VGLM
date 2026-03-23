#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
VisualGLM 带量化版
"""

import os
import platform
import torch
import argparse
import requests
import tempfile
import gc
import sys
import signal

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DEFAULT_MODEL_PATH = "./visualglm"


def download_image(url):
    try:
        print(f"  正在下载图片: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        ext = ".jpg"
        content_type = response.headers.get('content-type', '')
        if 'png' in content_type: ext = '.png'
        elif 'webp' in content_type: ext = '.webp'
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_file.write(response.content)
        temp_file.close()
        print(f"  ✓ 图片下载完成")
        return temp_file.name
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return None

def print_header():
    print("=" * 60)
    print("VGLM - 轻量化模型 (HuggingFace)")
    print("=" * 60)


def load_model(model_path, use_quant=False, quant_bits=4):
    """
    加载模型
    
    Args:
        model_path: 模型路径
        use_quant: 是否使用量化
        quant_bits: 量化位数 (4 或 8)
    """
    from transformers import AutoTokenizer, AutoModel
    
    print(f"\n[1/2] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✓ Tokenizer 加载完成")
    
    print(f"\n[2/2] 加载模型...")
    if use_quant:
        print(f"  使用 {quant_bits}-bit 量化")
    
    if use_quant and quant_bits == 4:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes as bnb
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="cuda" if torch.cuda.is_available() else "auto",
                low_cpu_mem_usage=True
            )
            
            for name, module in model.named_modules():
                if isinstance(module, bnb.nn.Linear4bit):
                    if not hasattr(module.weight, 'quant_state') or module.weight.quant_state is None:
                        module.weight = bnb.nn.Params4bit(
                            module.weight.data,
                            requires_grad=False,
                            quant_type="nf4"
                        ).cuda() if torch.cuda.is_available() else module.weight
            
            print("  ✓ 动态量化加载完成 (4-bit)")
            
        except ImportError:
            print("  ! 未检测到 bitsandbytes，正在自动安装...")
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes", "-q"])
            print("  ✓ bitsandbytes 安装完成，重新加载模型...")
            
            from transformers import BitsAndBytesConfig
            import bitsandbytes as bnb
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="cuda" if torch.cuda.is_available() else "auto",
                low_cpu_mem_usage=True
            )
            print("  ✓ 动态量化加载完成 (4-bit)")
            
        except Exception as e:
            print(f"  ! bitsandbytes 加载失败 ({e})，回退到 FP16...")
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            print("  ✓ FP16 加载完成")
    
    elif use_quant and quant_bits == 8:
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="cuda" if torch.cuda.is_available() else "auto",
                low_cpu_mem_usage=True
            )
            print("  ✓ 8-bit 量化加载完成")
            
        except Exception as e:
            print(f"  ! 8-bit 量化加载失败 ({e})，回退到 FP16...")
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            print("  ✓ FP16 加载完成")
    
    else:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        print("  ✓ FP16 加载完成")

    if hasattr(model, 'transformer') and hasattr(model.transformer, 'vision'):
        model.transformer.vision.to(torch.float16)
        
    model.eval()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        print(f"\nGPU 内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return model, tokenizer


def build_prompt(history, prefix):
    prompt = prefix
    for query, response in history:
        if isinstance(response, bytes):
            response = response.decode('utf-8', errors='replace')
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nVGLM：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif')

def is_image_input(text):
    """检测输入是否为图片路径或图片URL"""
    text = text.strip()
    if text.startswith(('http://', 'https://')):
        return True
    if any(text.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
        return True
    if os.path.isfile(text):
        return True
    return False

EXTYPE_PROMPTS = {
    'general': '描述这张图片',
    'background': '描述这张图片的背景环境',
    'detailed': '详细描述这张图片的所有内容',
    'english': 'Describe this image in English'
}

SYSTEM_PREFIX = "你是VGLM，一个图像描述模型。\n"
SYSTEM_PREFIX_EN = "You are VGLM, an image captioning model.\n"

def get_input(prompt_text):
    """更健壮的输入处理，处理终端编码问题"""
    sys.stdout.write(prompt_text)
    sys.stdout.flush()
    try:
        line = sys.stdin.buffer.readline()
        return line.decode('utf-8', errors='replace').strip()
    except:
        return input().strip()

#简单检测字符串是否主要为英文
def is_english_query(text):
    if not text:
        return False
    en_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    
    text_lower = text.lower()

    if "in chinese" in text_lower or "用中文" in text:
        return False
    if "in english" in text_lower or "用英文" in text:
        return True

    if "english" in text_lower:
        return True
    if "chinese" in text_lower:
        return False
    if "describe" in text_lower:
        return True
    if "描述" in text:
        return False
    
    return en_chars > zh_chars * 2


def safe_stream_chat(model, tokenizer, image_path, query, history):
    """带 UTF-8 容错的流式对话"""
    is_eng = is_english_query(query)
    prefix = SYSTEM_PREFIX_EN if is_eng else SYSTEM_PREFIX
    prompted_query = prefix + query
    
    for response, hist in model.stream_chat(tokenizer, image_path, prompted_query, history=history):
        if isinstance(response, bytes):
            response = response.decode('utf-8', errors='replace')
        elif isinstance(response, str):
            response = response.encode('utf-8', errors='replace').decode('utf-8', errors='replace')

        safe_hist = []
        for q, r in hist:
            if isinstance(r, bytes):
                r = r.decode('utf-8', errors='replace')
            elif isinstance(r, str):
                r = r.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            safe_hist.append((q, r))
        yield response, safe_hist, is_eng


def main():
    global stop_stream
    
    parser = argparse.ArgumentParser(description='VGLM 图像描述模型 (HuggingFace)')
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, 
                        help=f'模型路径 (默认: {DEFAULT_MODEL_PATH})')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=4, 
                        help='量化位数 (4 或 8，默认: 4)')
    parser.add_argument("--no-quant", action='store_true',
                        help='不使用量化（FP16，需要更多显存）')
    parser.add_argument("--prompt_zh", type=str, default="描述这张图片。", 
                        help='默认中文提示词')
    parser.add_argument("--prompt_en", type=str, default="Describe this image.", 
                        help='默认英文提示词')
    
    args = parser.parse_args()
    
    print_header()
    
    use_quant = not args.no_quant
    
    model, tokenizer = load_model(
        args.model_path, 
        use_quant=use_quant, 
        quant_bits=args.quant
    )
    
    print("\n" + "=" * 60)
    print("支持混合输入：输入中文回答中文，输入英文（如 Describe this image）回答英文。")
    print("提示词建议：")
    print("  1. '描述这张图片。' - 中文描述")
    print("  2. 'Describe this image.' - 英文描述")
    print("命令：clear 清空对话历史，stop 终止程序")
    print("=" * 60)
    
    os_name = platform.system()
    clear_command = 'cls' if os_name == 'Windows' else 'clear'
    stop_stream = False
    signal.signal(signal.SIGINT, signal_handler)
    
    while True:
        history = []
        prefix = "欢迎使用 VGLM 模型，输入图片路径和内容即可进行对话，clear 清空对话历史，stop 终止程序"
        print(prefix)
        image_input = get_input("\n请输入图片路径或URL：")
        if image_input.lower() == "stop":
            break
            
        temp_image_path = None
        if image_input.startswith(('http://', 'https://')):
            temp_image_path = download_image(image_input)
            if temp_image_path is None:
                continue
            image_path = temp_image_path
        else:
            image_path = image_input

        prefix = prefix + "\n" + image_input
        query = args.prompt_zh
        
        while True:
            try:
                count = 0
                with torch.no_grad(): 
                    for response, history, is_eng in safe_stream_chat(model, tokenizer, image_path, query, history=history):
                        if stop_stream:
                            stop_stream = False
                            break
                        count += 1
                        if count % 8 == 0:
                            os.system(clear_command)
                            print(build_prompt(history, prefix), flush=True)
                
                os.system(clear_command)
                print(build_prompt(history, prefix), flush=True)
                
                query = get_input("\n用户：")
                if query.lower() == "clear":
                    break
                if query.lower() == "stop":
                    stop_stream = True
                    exit(0)

                if is_image_input(query):
                    print(f"\n检测到新图片，正在切换...\n")
                    new_image_input = query.strip()
                    if new_image_input.startswith(('http://', 'https://')):
                        new_temp = download_image(new_image_input)
                        if new_temp:
                            if temp_image_path is not None and os.path.exists(temp_image_path):
                                try: os.unlink(temp_image_path)
                                except: pass
                            temp_image_path = new_temp
                            image_path = temp_image_path
                        else:
                            continue
                    else:
                        image_path = new_image_input
                        
                    prefix = prefix.split('\n')[0] + '\n' + image_path
                    query = args.prompt_zh
                    history = []  
                    
            except FileNotFoundError as e:
                print(f"错误：文件不存在 - {e}")
                print("请重新输入正确的图片路径。")
                image_path = get_input("请输入图片路径：")
                if image_path.lower() == "stop":
                    print("程序终止。")
                    exit(0)
                prefix = prefix.split('\n')[0] + '\n' + image_path
                query = args.prompt_zh
                history = []
            except UnicodeDecodeError as e:
                print(f"编码错误（已自动跳过）：{e}")
                query = get_input("\n用户：")
                if query.lower() == "clear":
                    break
                if query.lower() == "stop":
                    stop_stream = True
                    exit(0)
            except Exception as e:
                print(f"错误：{e}")
                query = get_input("\n用户：")
                if query.lower() == "clear":
                    break
                if query.lower() == "stop":
                    stop_stream = True
                    exit(0)


if __name__ == "__main__":
    main()
