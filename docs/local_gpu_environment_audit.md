# GeoAgri-AI: Local GPU Hardware & Software Environment Audit Report

**Document ID:** `docs/local_gpu_environment_audit.md`  
**Auditor:** Senior MLOps Engineer & Systems Auditor  
**Date:** October 28, 2026  
**Status:** Completed (GPU Hardware Verified | PyTorch CUDA Blocker Identified)

---

## 1. Executive Summary

A diagnostic audit of the local computing hardware, operating system, graphics driver, and Python environments was conducted to evaluate readiness for local YOLO training.

* **NVIDIA Hardware Verified:** The machine possesses an **NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)** with NVIDIA Driver **581.86** supporting up to CUDA **13.0**. Current VRAM utilization is minimal ($< 90\ \text{MB}$ used, $> 6,000\ \text{MB}$ available).
* **Available Project Disk Space:** 57.03 GB free on drive `C:`.
* **Primary Blocker Identified:**
  - In `GeoAgri-AI\.venv`: PyTorch and Ultralytics are **not installed**.
  - In `Archaeological-Site-Mapping-AI-V2\.venv`: PyTorch is installed as **`2.7.1+cpu` (CPU ONLY, `torch.cuda.is_available() == False`)**.
  - *Distinction:* While XGBoost successfully uses CUDA (`device="cuda"`) through its own built-in runtime, YOLO requires **PyTorch with CUDA support** to utilize the GPU.
* **Readiness Verdict:** **LOCAL GPU TRAINING IS BLOCKED UNTIL PYTORCH CUDA WHEELS ARE INSTALLED.**

---

## 2. Hardware Diagnostics (`nvidia-smi`)

```
NVIDIA-SMI 581.86              Driver Version: 581.86          CUDA Version: 13.0
GPU Name:                      NVIDIA GeForce RTX 3050 Laptop GPU
Bus ID:                        00000000:01:00.0
Total Physical VRAM:           6,144 MiB (6.00 GB)
Current VRAM in Use:           81 MiB
Free VRAM Available:           6,063 MiB
Operating Temperature:         48°C (Idle)
Processor:                     Intel Core i5-12450HX (8 Cores, 12 Threads)
Operating System:              Windows 11 Home (64-bit, Build 26300)
Storage Available (C:):        57.03 GB Free / 474.72 GB Total
```

---

## 3. Environment Inventory & Package Audit

Three distinct Python virtual environments were located on disk:

| Environment Path | Python Version | PyTorch Status | Torchvision | Ultralytics | OpenCV | XGBoost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`GeoAgri-AI\.venv`** *(Active Repo)* | 3.13.3 | **NOT INSTALLED** | NOT INSTALLED | NOT INSTALLED | 5.0.0 | 3.4.1 (CUDA OK) |
| **`..\.venv`** *(Parent Root)* | 3.13.3 | **NOT INSTALLED** | NOT INSTALLED | NOT INSTALLED | NOT INSTALLED | NOT INSTALLED |
| **`..\Archaeological-Site-Mapping-AI-V2\...`** | 3.13.3 | **`2.7.1+cpu` (CUDA: False)** | 0.22.1+cpu | **8.4.24** | 4.13.0 | 3.2.0 (CUDA OK) |

### Diagnostic Output from Inspection Script:
```
=== Environment: GeoAgri-AI .venv ===
Path: C:\...\GeoAgri-AI\.venv\Scripts\python.exe
  Python Version: 3.13.3
  torch: NOT INSTALLED
  ultralytics: NOT INSTALLED

=== Environment: Arch-Site-Mapping-AI-V2 .venv ===
Path: C:\...\Archaeological-Site-Mapping-AI-V2\.venv\Scripts\python.exe
  Python Version: 3.13.3
  torch: 2.7.1+cpu (CUDA: False)
  torchvision: 0.22.1+cpu
  ultralytics: 8.4.24
```

---

## 4. Root Cause Analysis: CPU-Only PyTorch

When dependencies were previously installed via standard `pip install torch torchvision` on Windows without specifying an explicit PyTorch CUDA index, PyPI served the default CPU wheel (`+cpu`).

Because `torch.cuda.is_available()` returns `False`:
* Any YOLO training launched with `device=0` or `device="cuda"` will immediately crash with:
  `RuntimeError: Torch not compiled with CUDA enabled` or fall back to excruciatingly slow CPU execution.

---

## 5. Python 3.13 Compatibility Considerations

* **Python Version:** 3.13.3.
* **Ultralytics Compatibility:** Ultralytics version `8.4.24` is fully compatible with Python 3.13.
* **PyTorch CUDA Wheels for Python 3.13:** Official PyTorch release wheels for CUDA on Python 3.13 (`torch>=2.6.0`, CUDA 12.4 / 12.6) are available on `download.pytorch.org/whl/cu124` or `download.pytorch.org/whl/cu126`.

---

## 6. Recommended Remediation & Setup Plan

To enable local GPU training on the RTX 3050 without altering any existing production code:

### Step 1: Install PyTorch with CUDA in Target Virtual Environment
```powershell
# In the chosen virtual environment:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Step 2: Install Ultralytics
```powershell
pip install ultralytics==8.4.24
```

### Step 3: Run Diagnostic Smoke Test
```python
import torch
assert torch.cuda.is_available(), "CUDA still not available!"
print(f"CUDA Verified: {torch.cuda.get_device_name(0)}")
t = torch.randn((1000, 1000), device="cuda") @ torch.randn((1000, 1000), device="cuda")
print(f"Tensor calculation successful on {t.device}")
```

*In compliance with Safety Rule #5, these packages have **not** been installed during this audit.*

