#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
VisualGLM 带量化版（轻量化本地运行）
"""

import os
import platform
import signal
import gc
import torch

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

gc.collect()

print("=" * 60)
print("VGLM - 轻量化模型")
print("=" * 60)

offload_dir = "/tmp/visualglm_offload"
os.makedirs(offload_dir, exist_ok=True)

print("\n[1/3] 加载 Tokenizer...")
from transformers import AutoTokenizer
# 请修改为你自己的 visualglm 基础模型路径
tokenizer = AutoTokenizer.from_pretrained(
    "/gemini/code/VGLM/visualglm",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
print("✓ Tokenizer 加载完成")

print("\n[2/3] 加载模型...")

from transformers import AutoModel

def _load_model_bnb():
    """使用 bitsandbytes 进行流式 4-bit 量化加载"""
    from transformers import BitsAndBytesConfig
    import bitsandbytes as bnb
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # 请修改为你自己的 visualglm 基础模型路径
    _model = AutoModel.from_pretrained(
        "/gemini/code/VGLM/visualglm",
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="cuda",
        low_cpu_mem_usage=True
    )

    for name, module in _model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            if not hasattr(module.weight, 'quant_state') or module.weight.quant_state is None:
                module.weight = bnb.nn.Params4bit(
                    module.weight.data,
                    requires_grad=False,
                    quant_type="nf4"
                ).cuda()
    
    _model.eval()
    return _model


try:
    print("  尝试使用 bitsandbytes 进行流式 4-bit 量化加载 ...")
    model = _load_model_bnb()
    print("  ✓ 动态量化加载完成 (4-bit)")
    
except ImportError:
    print("  ! 未检测到 bitsandbytes，正在自动安装...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes", "-q"])
    print("  ✓ bitsandbytes 安装完成，重新加载模型...")
    model = _load_model_bnb()
    print("  ✓ 动态量化加载完成 (4-bit)")

except Exception as e:
    print(f"  ! bitsandbytes 加载失败 ({e})，回退到传统方式...")
    # 请修改为你自己的 visualglm 基础模型路径
    model = AutoModel.from_pretrained(
        "/gemini/code/VGLM/visualglm",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    print("  传统原生量化模型...")
    model = model.quantize(4)
    model.eval()
    print("  ✓ 传统量化完成")

gc.collect()
torch.cuda.empty_cache()

if torch.cuda.is_available():
    print(f"\nGPU 内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

print("\n" + "=" * 60)
print("准备就绪！")
print("=" * 60)

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


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


SYSTEM_PREFIX = "你是VGLM，一个轻量化图像描述模型。\n"

def safe_stream_chat(model, tokenizer, image_path, query, history):
    """带 UTF-8 容错的流式对话"""
    prompted_query = SYSTEM_PREFIX + query if not history else query
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
        yield response, safe_hist


def main():
    global stop_stream
    signal.signal(signal.SIGINT, signal_handler)
    
    while True:
        history = []
        prefix = "欢迎使用 VGLM 模型，输入图片路径和内容即可进行对话，clear 清空对话历史，stop 终止程序"
        print(prefix)
        image_path = input("\n请输入图片路径：")
        if image_path == "stop":
            break
        prefix = prefix + "\n" + image_path
        query = "描述这张图片。"
        
        while True:
            try:
                count = 0
                with torch.no_grad(): 
                    for response, history in safe_stream_chat(model, tokenizer, image_path, query, history=history):
                        if stop_stream:
                            stop_stream = False
                            break
                        count += 1
                        if count % 8 == 0:
                            os.system(clear_command)
                            print(build_prompt(history, prefix), flush=True)
                
                os.system(clear_command)
                print(build_prompt(history, prefix), flush=True)
                
                query = input("\n用户：")
                if query.strip() == "clear":
                    break
                if query.strip() == "stop":
                    stop_stream = True
                    exit(0)

                if is_image_input(query):
                    print(f"\n检测到新图片，正在切换...\n")
                    image_path = query.strip()
                    prefix = prefix.split('\n')[0] + '\n' + image_path
                    query = "描述这张图片。"
                    history = []  
                    
            except FileNotFoundError as e:
                print(f"错误：文件不存在 - {e}")
                print("请重新输入正确的图片路径。")
                image_path = input("请输入图片路径：")
                if image_path == "stop":
                    print("程序终止。")
                    exit(0)
                prefix = prefix.split('\n')[0] + '\n' + image_path
                query = "描述这张图片。"
                history = []
            except UnicodeDecodeError as e:
                print(f"编码错误（已自动跳过）：{e}")
                query = input("\n用户：")
                if query.strip() == "clear":
                    break
                if query.strip() == "stop":
                    stop_stream = True
                    exit(0)
            except Exception as e:
                print(f"错误：{e}")
                query = input("\n用户：")
                if query.strip() == "clear":
                    break
                if query.strip() == "stop":
                    stop_stream = True
                    exit(0)


if __name__ == "__main__":
    main()
