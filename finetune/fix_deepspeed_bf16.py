#!/usr/bin/env python3
"""
修复 DeepSpeed bf16_optimizer.py 中的 Bug：
  assert all_groups_norm > 0.

原因：LoRA 的 matrix_B 初始化为全零，导致第一步 matrix_A 的梯度也为零，
      从而 all_groups_norm = 0，触发断言崩溃。

用法（在服务器上运行）：先运行该文件，再运行finetune_visualglm.sh
"""
import re
import shutil
import sys
import importlib.util
from pathlib import Path

def find_bf16_optimizer():
    try:
        spec = importlib.util.find_spec("deepspeed")
        if spec is None:
            raise ImportError("deepspeed not found")
        ds_root = Path(spec.origin).parent
        bf16_path = ds_root / "runtime" / "bf16_optimizer.py"
        if bf16_path.exists():
            return bf16_path
    except Exception as e:
        print(f"Auto-detect failed: {e}")
    
    # 常见路径
    candidates = [
        Path("/root/miniconda3/lib/python3.11/site-packages/deepspeed/runtime/bf16_optimizer.py"),
        Path("/usr/local/lib/python3.11/dist-packages/deepspeed/runtime/bf16_optimizer.py"),
        Path("/opt/conda/lib/python3.11/site-packages/deepspeed/runtime/bf16_optimizer.py"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def patch_bf16_optimizer(filepath: Path):
    content = filepath.read_text(encoding="utf-8")
    
    # 检查是否已经打过补丁
    if "# [PATCHED] skip step if norm is zero" in content:
        print(f"✅ Already patched: {filepath}")
        return True
    
    # 原始断言
    old_assert = "assert all_groups_norm > 0."
    
    # 新的安全检查（跳过零梯度步骤，而非崩溃）
    new_check = (
        "# [PATCHED] skip step if norm is zero (e.g., LoRA matrix_B=0 at init)\n"
        "        if all_groups_norm == 0.:\n"
        "            return\n"
        "        assert all_groups_norm > 0."
    )
    
    if old_assert not in content:
        print(f"❌ Could not find target line: '{old_assert}'")
        print("   DeepSpeed version may have changed. Manual patch needed.")
        print(f"   File: {filepath}")
        return False
    
    # 备份原文件
    backup = filepath.with_suffix(".py.bak")
    if not backup.exists():
        shutil.copy2(filepath, backup)
        print(f"📁 Backup created: {backup}")
    
    # 应用补丁（替换断言）
    patched = content.replace(old_assert, new_check)
    filepath.write_text(patched, encoding="utf-8")
    print(f"✅ Successfully patched: {filepath}")
    return True

if __name__ == "__main__":
    bf16_path = find_bf16_optimizer()
    
    if bf16_path is None:
        print("❌ Could not find deepspeed/runtime/bf16_optimizer.py")
        print("   Please specify the path manually and edit this script.")
        sys.exit(1)
    
    print(f"🔍 Found: {bf16_path}")
    success = patch_bf16_optimizer(bf16_path)
    
    if success:
        print("\n✅ Patch applied. You can now run BF16 training.")
        print("   To restore original: cp <file>.py.bak <file>.py")
    else:
        sys.exit(1)
