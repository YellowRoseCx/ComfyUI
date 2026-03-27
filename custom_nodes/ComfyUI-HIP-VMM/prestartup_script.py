import os
import ctypes
import torch

def prestartup():
    so_path = os.path.join(os.path.dirname(__file__), 'hip_vmm_allocator.so')
    if not os.path.exists(so_path):
        print(f"[HIP VMM WARNING] Could not find {so_path}. You must run build.sh first.")
        return

    try:
        # We load the global library and set up the limits types early
        lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
        lib.update_allocator_limits.argtypes = [ctypes.c_size_t, ctypes.c_size_t]

        # In ComfyUI prestartup context, this runs right before PyTorch memory is utilized.
        # This is absolutely critical because torch.cuda.memory.change_current_allocator
        # raises a fatal RuntimeError if called after any memory allocation.
        new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
            so_path,
            alloc_fn_name='my_malloc',
            free_fn_name='my_free'
        )
        torch.cuda.memory.change_current_allocator(new_alloc)

        print("[HIP VMM] Custom Pluggable Allocator successfully hooked during ComfyUI prestartup.")
    except Exception as e:
        print(f"[HIP VMM ERROR] Prestartup failed to inject pluggable allocator from {so_path}: {e}")

prestartup()
