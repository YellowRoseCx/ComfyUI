import os
import sys
import ctypes

try:
    import torch
except ImportError:
    print("[HIP VMM ERROR] PyTorch is not installed. Please install torch before running this script.")
    sys.exit(1)

# Enable AMD ROCm large pinned memory allocations and system fallbacks
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

# CRITICAL: AMD's HSA driver strictly limits pinned memory (Host RAM for GPU) to 3/8 of
# total system RAM by default (e.g. 24GB out of 64GB). Because RDNA2 VMM requires pinned memory
# for the Host fallback, massive allocations like a 3GB VAE will easily exceed this artificial
# driver limit. This forces the HSA driver to allow up to 95% of system RAM to be pinned.
# NOTE: Your OS must still allow this via `ulimit -l unlimited`
os.environ["GPU_MAX_HOST_MEM"] = "95"

# Locate the compiled shared library
so_path = os.path.join(os.path.dirname(__file__), 'custom_nodes', 'ComfyUI-HIP-VMM', 'hip_vmm_allocator.so')

if not os.path.exists(so_path):
    print(f"[HIP VMM ERROR] Compiled allocator library not found at: {so_path}")
    print("Please run the build.sh script inside custom_nodes/ComfyUI-HIP-VMM/ first.")
    sys.exit(1)

print("[HIP VMM] Initializing Custom PyTorch Pluggable Allocator for ROCm...")

try:
    # We must load the library into the global namespace so Python ctypes can access the exported limits function later
    lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    lib.update_allocator_limits.argtypes = [ctypes.c_size_t, ctypes.c_size_t]

    # Crucially, this must be executed BEFORE ANY other PyTorch CUDA function or ComfyUI module is imported
    # because PyTorch locks the caching allocator as soon as it initializes the GPU context.
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        so_path,
        alloc_fn_name='my_malloc',
        free_fn_name='my_free'
    )
    torch.cuda.memory.change_current_allocator(new_alloc)

    print("[HIP VMM] Allocator successfully injected and locked.")

    # CRITICAL FIX: PyTorch's CUDAPluggableAllocator does not implement memory statistics.
    # When ComfyUI probes memory_stats() or memory_allocated(), the native PyTorch C++ extension
    # throws a RuntimeError: CUDAPluggableAllocator does not yet support getDeviceStats.
    # To prevent ComfyUI from crashing during boot or model loading, we monkey-patch these functions
    # to return zero/dummy values, as our C++ allocator manages OOM at the hardware level natively.
    def mock_memory_stats(device=None):
        # Supply exactly the keys ComfyUI reads to avoid KeyErrors
        return {
            'allocated_bytes.all.current': 0,
            'reserved_bytes.all.current': 0,
            'active_bytes.all.current': 0,
            'inactive_split_bytes.all.current': 0
        }
    def mock_memory_stats_as_nested_dict(device=None):
        return {
            'allocated_bytes': {'all': {'current': 0}},
            'reserved_bytes': {'all': {'current': 0}},
            'active_bytes': {'all': {'current': 0}},
            'inactive_split_bytes': {'all': {'current': 0}}
        }
    def mock_memory_allocated(device=None):
        return 0
    def mock_max_memory_allocated(device=None):
        return 0
    def mock_memory_reserved(device=None):
        return 0
    def mock_max_memory_reserved(device=None):
        return 0

    torch.cuda.memory_stats = mock_memory_stats
    torch.cuda.memory_stats_as_nested_dict = mock_memory_stats_as_nested_dict
    torch.cuda.memory_allocated = mock_memory_allocated
    torch.cuda.max_memory_allocated = mock_max_memory_allocated
    torch.cuda.memory_reserved = mock_memory_reserved
    torch.cuda.max_memory_reserved = mock_max_memory_reserved

    print("[HIP VMM] PyTorch memory stat functions successfully mocked for ComfyUI compatibility.")

except Exception as e:
    print(f"[HIP VMM FATAL] Failed to inject custom pluggable allocator: {e}")
    sys.exit(1)

# Now that the allocator is safely locked, we can hand execution over to ComfyUI natively.
print("[HIP VMM] Handing execution to ComfyUI main...")

import main

if __name__ == "__main__":
    # Imitate ComfyUI's standard launch process
    event_loop, _, start_all_func = main.start_comfyui()
    try:
        x = start_all_func()
        import app.logger
        app.logger.print_startup_warnings()
        event_loop.run_until_complete(x)
    except KeyboardInterrupt:
        import logging
        logging.info("\nStopped server")
    finally:
        try:
            # We safely access the module if it was loaded during ComfyUI's main run
            # instead of explicitly hard-importing it and risking a ModuleNotFoundError
            # if the PYTHONPATH isn't perfectly structured or if the user is running a lighter fork
            import sys
            if 'app.assets.seeder' in sys.modules:
                sys.modules['app.assets.seeder'].asset_seeder.shutdown()
        except Exception:
            pass
        main.cleanup_temp()
