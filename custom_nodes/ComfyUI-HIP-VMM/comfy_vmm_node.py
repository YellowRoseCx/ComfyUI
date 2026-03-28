import os
import ctypes

# Universal any type so the node can accept any kind of link in ComfyUI
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

# instantiate
any_typ = AnyType("*")

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
            },
            "optional": {
                # A universal passthrough to force execution order in ComfyUI
                "any_input": (any_typ, ),
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("any_output",)
    FUNCTION = "apply_limits"
    CATEGORY = "advanced/memory"
    # Still an output node in case the user just drops it in the graph disconnected
    OUTPUT_NODE = True

    def apply_limits(self, vram_limit_mb, ram_limit_mb, any_input=None):
        # We can dynamically grab the preloaded library from RTLD_GLOBAL
        so_path = os.path.join(os.path.dirname(__file__), 'hip_vmm_allocator.so')
        try:
            lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            lib.update_allocator_limits.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
            lib.update_allocator_limits(vram_limit_mb, ram_limit_mb)
        except Exception as e:
            print(f"[HIP VMM WARNING] Cannot update limits. Make sure you compiled the library and set PYTORCH_CUDA_ALLOC_CONF. {e}")

        # Return the input exactly as we received it to allow chaining
        return (any_input,)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "ComfyHIPVMMNode": ComfyHIPVMMNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyHIPVMMNode": "HIP VMM Allocator Limits"
}