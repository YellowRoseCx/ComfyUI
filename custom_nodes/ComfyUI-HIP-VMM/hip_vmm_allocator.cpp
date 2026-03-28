#include <hip/hip_runtime.h>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <cstddef>

// Typedef cudaStream_t to hipStream_t so we don't need to include massive PyTorch C++ headers
// PyTorch's change_current_allocator uses cudaStream_t in the C API footprint.
typedef hipStream_t cudaStream_t;

// PyTorch expects ssize_t for size in allocator signatures
typedef ptrdiff_t ssize_t;

// Global Variables & Thresholds
static std::atomic<size_t> g_vram_limit_bytes(0);
static std::atomic<size_t> g_ram_limit_bytes(0);

// Tracks currently allocated bytes
static std::atomic<size_t> g_current_device_allocated(0);
static std::atomic<size_t> g_current_host_allocated(0);

// State tracking for logging
static std::atomic<bool> g_fallback_active(false);

// Mutex for map operations
static std::mutex g_alloc_mutex;

// We need to keep track of allocation metadata to cleanly unmap and release
struct AllocationMeta {
    size_t padded_size;
    hipMemGenericAllocationHandle_t handle;
    bool is_host_fallback;
    bool used_unified_memory;
};

// Map original returned pointer to allocation metadata
static std::unordered_map<void*, AllocationMeta> g_allocations;

// Utility for HIP Error Checking
#define HIP_CHECK(status) \
    if (status != hipSuccess) { \
        std::cerr << "[HIP VMM ERROR] " << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return nullptr; \
    }

// Utility for HIP Error Checking in void functions
// We deliberately DO NOT return here so that cleanup sequences can continue
#define HIP_LOG_ERROR(status, msg) \
    if (status != hipSuccess) { \
        std::cerr << "[HIP VMM ERROR] " << msg << ": " << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

extern "C" {

// Update allocator limits dynamically via Python ctypes
void update_allocator_limits(size_t vram_mb, size_t ram_mb) {
    g_vram_limit_bytes = vram_mb * 1024 * 1024;
    g_ram_limit_bytes = ram_mb * 1024 * 1024;
    std::cout << "[HIP VMM] Limits updated: VRAM=" << vram_mb << " MB, RAM=" << ram_mb << " MB" << std::endl;
}

// PyTorch Pluggable Allocator signature
void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
    if (size <= 0) return nullptr;

    static std::atomic<bool> g_first_alloc(true);
    if (g_first_alloc.exchange(false)) {
        std::cout << "[HIP VMM] Verified: Custom my_malloc intercept successfully reached." << std::endl;
    }

    // 1. Get Allocation Granularity for ROCm 7.x
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    // Set initially to device
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;

    size_t granularity = 0;
    hipError_t status = hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum);
    if (status != hipSuccess) {
        std::cerr << "[HIP VMM ERROR] hipMemGetAllocationGranularity failed: " << hipGetErrorString(status) << std::endl;
        return nullptr;
    }

    // 2. Pad size mathematically
    size_t padded_size = ((size + granularity - 1) / granularity) * granularity;

    void* ptr = nullptr;
    hipMemGenericAllocationHandle_t handle = {};
    bool is_host_fallback = false;
    bool used_unified_memory = false;

    // 3. Check VRAM Limits
    if (g_vram_limit_bytes > 0 && (g_current_device_allocated + padded_size > g_vram_limit_bytes)) {
        // Exceeds VRAM limit naturally, skip VMM and directly use Unified Memory Fallback
        status = hipErrorOutOfMemory;
    } else {
        // Attempt Native VMM Device Allocation
        status = hipMemAddressReserve(&ptr, padded_size, 0, nullptr, 0);
        if (status == hipSuccess) {
            status = hipMemCreate(&handle, padded_size, &prop, 0);

            if (status == hipSuccess) {
                // Map Physical Memory to Virtual Address
                status = hipMemMap(ptr, padded_size, 0, handle, 0);
                if (status == hipSuccess) {
                    // Grant Access
                    hipMemAccessDesc accessDesc = {};
                    accessDesc.location.type = hipMemLocationTypeDevice;
                    accessDesc.location.id = device;
                    accessDesc.flags = hipMemAccessFlagsProtReadWrite;

                    status = hipMemSetAccess(ptr, padded_size, &accessDesc, 1);
                    if (status != hipSuccess) {
                        hipMemUnmap(ptr, padded_size);
                        hipMemRelease(handle);
                        hipMemAddressFree(ptr, padded_size);
                    }
                } else {
                    hipMemRelease(handle);
                    hipMemAddressFree(ptr, padded_size);
                }
            } else {
                hipMemAddressFree(ptr, padded_size);
            }
        }
    }

    // 4. The Fallback (Unified Memory System RAM)
    if (status != hipSuccess) {
        if (!g_fallback_active.exchange(true)) {
             std::cout << "[HIP VMM WARN] VRAM limit reached or VMM allocation failed, falling back to Unified Memory (Host RAM over PCIe)" << std::endl;
        }

        // Verify Host allocation doesn't exceed RAM limit
        if (g_ram_limit_bytes > 0 && (g_current_host_allocated + padded_size > g_ram_limit_bytes)) {
            std::cerr << "[HIP VMM ERROR] Host RAM limit exceeded! Attempted to allocate " << padded_size << " bytes." << std::endl;
            return nullptr;
        }

        // Use standard Managed Memory which dynamically pages to CPU DDR4/DDR5
        // without fighting the VMM pinned-memory size constraints of the HSA driver
        status = hipMallocManaged(&ptr, padded_size, hipMemAttachGlobal);
        if (status != hipSuccess) {
            std::cerr << "[HIP VMM FATAL] Unified Memory (Host) allocation failed: " << hipGetErrorString(status) << std::endl;
            std::cerr << "[HIP VMM HINT] This is often caused by the OS out of physical RAM and swap space." << std::endl;
            return nullptr;
        }

        is_host_fallback = true;
        used_unified_memory = true;
    } else {
        // Device allocation succeeded, we are not in fallback for this allocation
        if (g_fallback_active.load() && g_current_device_allocated == 0) {
            // Optional: reset fallback log when memory drops
            g_fallback_active.store(false);
        }
    }

    // Update Trackers
    if (is_host_fallback) {
        g_current_host_allocated += padded_size;
    } else {
        g_current_device_allocated += padded_size;
    }

    // Save metadata
    {
        std::lock_guard<std::mutex> lock(g_alloc_mutex);
        g_allocations[ptr] = {padded_size, handle, is_host_fallback, used_unified_memory};
    }

    return ptr;
}

// PyTorch Pluggable Allocator signature
void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    if (!ptr) return;

    AllocationMeta meta;
    {
        std::lock_guard<std::mutex> lock(g_alloc_mutex);
        auto it = g_allocations.find(ptr);
        if (it == g_allocations.end()) {
            std::cerr << "[HIP VMM ERROR] Attempted to free untracked pointer: " << ptr << std::endl;
            return;
        }
        meta = it->second;
        g_allocations.erase(it);
    }

    // Update trackers before cleanup to avoid race states where limits appear full
    if (meta.is_host_fallback) {
        g_current_host_allocated -= meta.padded_size;
    } else {
        g_current_device_allocated -= meta.padded_size;

        // Reset the warning log if we drop below the VRAM limit.
        // This ensures the user is warned during each separate massive generation spike,
        // rather than just once during the entire ComfyUI session lifecycle.
        if (g_fallback_active.load() && g_current_device_allocated < g_vram_limit_bytes && g_current_host_allocated == 0) {
            g_fallback_active.store(false);
        }
    }

    // Cleanup strict order based on allocation type:
    hipError_t status;
    if (meta.used_unified_memory) {
        // Simple free for hipMallocManaged
        status = hipFree(ptr);
        HIP_LOG_ERROR(status, "hipFree failed on Unified Memory");
    } else {
        // VMM Cleanup strict order:
        // 1. Unmap
        // 2. Release Handle
        // 3. Free Address
        // CRITICAL: We must attempt to execute all three even if one fails to prevent catastrophic leaks.

        status = hipMemUnmap(ptr, meta.padded_size);
        HIP_LOG_ERROR(status, "hipMemUnmap failed");

        status = hipMemRelease(meta.handle);
        HIP_LOG_ERROR(status, "hipMemRelease failed");

        status = hipMemAddressFree(ptr, meta.padded_size);
        HIP_LOG_ERROR(status, "hipMemAddressFree failed");
    }
}

} // extern "C"
