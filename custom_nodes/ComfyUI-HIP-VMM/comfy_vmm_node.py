import os
import ctypes

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
        # We can dynamically grab the preloaded library from RTLD_GLOBAL
        so_path = os.path.join(os.path.dirname(__file__), 'hip_vmm_allocator.so')
        try:
            lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            lib.update_allocator_limits(vram_limit_mb, ram_limit_mb)
        except Exception as e:
            print(f"[HIP VMM WARNING] Cannot update limits. Allocator was not successfully loaded. {e}")
        return ()

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "ComfyHIPVMMNode": ComfyHIPVMMNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyHIPVMMNode": "HIP VMM Allocator Limits"
}
