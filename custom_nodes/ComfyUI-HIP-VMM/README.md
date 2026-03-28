# 🚀 PyTorch HIP VMM Pluggable Allocator

A custom PyTorch Pluggable Memory Allocator written in C++ targeting AMD RDNA2 architectures (specifically `gfx1030` like the RX 6800 XT) running ROCm 7.12.0.

This allocator gracefully prevents `hipErrorOutOfMemory` crashes by dynamically redirecting PyTorch memory allocations directly to Host Memory (System RAM) over the PCIe bus when the VRAM limit is reached. It exposes ComfyUI nodes for real-time limit thresholding.

## How to Install and Compile

1. Run `build.sh` to compile the library:
```bash
cd custom_nodes/ComfyUI-HIP-VMM
bash build.sh
```
This will generate `hip_vmm_allocator.so`.

## CRITICAL: How to Launch ComfyUI
Because PyTorch locks its memory allocator extremely early in the boot cycle, you **cannot** inject custom allocators purely from Python inside a ComfyUI custom node without throwing a `RuntimeError`.

The native PyTorch `PYTORCH_CUDA_ALLOC_CONF` environment variable string parser does not cleanly support `backend:pluggable` natively without crashing in `torch.cuda.memory`. Therefore, you must use the provided wrapper script to inject the allocator instantly *before* ComfyUI loads any deep PyTorch context.

From the root ComfyUI directory, run:

```bash
python launch_vmm.py
```
This script acts exactly like `main.py` but intercepts the CUDA context to install the HIP VMM APIs cleanly.

## Usage
Add the **HIP VMM Allocator Limits** node to your ComfyUI workflow. Connect its inputs to set the `vram_limit_mb` and `ram_limit_mb`. Because it has no graphical outputs, ensure it runs at least once by having it trigger on a required step. The limits update dynamically.