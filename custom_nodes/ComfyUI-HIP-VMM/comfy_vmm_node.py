import os
import ctypes
import torch

# Global tracker for the loaded allocator to ensure it is initialized at import
_VMM_ALLOCATOR_LOADED = False
_VMM_LIB = None

def _initialize_allocator():
    global _VMM_ALLOCATOR_LOADED, _VMM_LIB
    so_path = os.path.join(os.path.dirname(__file__), 'hip_vmm_allocator.so')
    if not os.path.exists(so_path):
        print(f"[HIP VMM WARNING] Could not find {so_path}. You must run build.sh first.")
        return

    try:
        # Load the allocator library
        _VMM_LIB = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
        _VMM_LIB.update_allocator_limits.argtypes = [ctypes.c_size_t, ctypes.c_size_t]

        # PyTorch Pluggable Allocator API initialization
        new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
            so_path,
            alloc_fn_name='my_malloc',
            free_fn_name='my_free'
        )
        torch.cuda.memory.change_current_allocator(new_alloc)

        _VMM_ALLOCATOR_LOADED = True
        print("[HIP VMM] Custom Pluggable Allocator initialized and set as current.")
    except Exception as e:
        print(f"[HIP VMM ERROR] Failed to load allocator from {so_path}: {e}")

# Trigger initialization immediately on module import
_initialize_allocator()

class ComfyHIPVMMNode:
    """
    ComfyUI Node to configure the custom PyTorch Pluggable Memory Allocator
    limits dynamically during execution using AMD HIP VMM APIs.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vram_limit_mb": ("INT", {
                    "default": 14336, # Example default: 14GB for 16GB cards
                    "min": 1024,
                    "max": 131072, # Arbitrary large number
                    "step": 256
                }),
                "ram_limit_mb": ("INT", {
                    "default": 32768, # Default: 32GB system RAM fallback limit
                    "min": 1024,
                    "max": 262144, # 256GB
                    "step": 1024
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "apply_limits"
    CATEGORY = "advanced/memory"
    # REQUIRED to force ComfyUI to actually run the node since it has no outputs
    OUTPUT_NODE = True

    def apply_limits(self, vram_limit_mb, ram_limit_mb):
        global _VMM_ALLOCATOR_LOADED, _VMM_LIB
        if _VMM_ALLOCATOR_LOADED and _VMM_LIB is not None:
            _VMM_LIB.update_allocator_limits(vram_limit_mb, ram_limit_mb)
        else:
            print("[HIP VMM WARNING] Cannot update limits. Allocator was not successfully loaded.")
        return ()

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "ComfyHIPVMMNode": ComfyHIPVMMNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyHIPVMMNode": "HIP VMM Allocator Limits"
}
