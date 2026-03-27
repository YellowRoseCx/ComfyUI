#include <hip/hip_runtime.h>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstdint>

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

    // 3. Reserve Virtual Address Space
    void* ptr = nullptr;
    HIP_CHECK(hipMemAddressReserve(&ptr, padded_size, 0, nullptr, 0));

    hipMemGenericAllocationHandle_t handle;
    bool is_host_fallback = false;

    // 4. Attempt Device Allocation
    if (g_vram_limit_bytes > 0 && (g_current_device_allocated + padded_size > g_vram_limit_bytes)) {
        // Exceeds VRAM limit natively, skip directly to fallback
        status = hipErrorOutOfMemory;
    } else {
        status = hipMemCreate(&handle, padded_size, &prop, 0);
    }

    // 5. The Fallback (System RAM)
    if (status != hipSuccess) {
        if (!g_fallback_active.exchange(true)) {
             std::cout << "[HIP VMM WARN] VRAM limit reached, falling back to Host RAM over PCIe" << std::endl;
        }

        // Switch location to Host
        prop.location.type = hipMemLocationTypeHost;
        // Host id is typically 0, but can just let it be since it's Host
        prop.location.id = 0;

        // Verify Host allocation doesn't exceed RAM limit
        if (g_ram_limit_bytes > 0 && (g_current_host_allocated + padded_size > g_ram_limit_bytes)) {
            std::cerr << "[HIP VMM ERROR] Host RAM limit exceeded! Attempted to allocate " << padded_size << " bytes." << std::endl;
            hipMemAddressFree(ptr, padded_size);
            return nullptr;
        }

        status = hipMemCreate(&handle, padded_size, &prop, 0);
        if (status != hipSuccess) {
            std::cerr << "[HIP VMM FATAL] Host memory allocation failed: " << hipGetErrorString(status) << std::endl;
            hipMemAddressFree(ptr, padded_size);
            return nullptr;
        }
        is_host_fallback = true;
    } else {
        // Device allocation succeeded, we are not in fallback for this allocation
        if (g_fallback_active.load() && g_current_device_allocated == 0) {
            // Optional: reset fallback log when memory drops
            g_fallback_active.store(false);
        }
    }

    // 6. Map Physical Memory to Virtual Address
    status = hipMemMap(ptr, padded_size, 0, handle, 0);
    if (status != hipSuccess) {
        std::cerr << "[HIP VMM ERROR] hipMemMap failed: " << hipGetErrorString(status) << std::endl;
        hipMemRelease(handle);
        hipMemAddressFree(ptr, padded_size);
        return nullptr;
    }

    // 7. Grant Access
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = device;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;

    status = hipMemSetAccess(ptr, padded_size, &accessDesc, 1);
    if (status != hipSuccess) {
        std::cerr << "[HIP VMM ERROR] hipMemSetAccess failed: " << hipGetErrorString(status) << std::endl;
        // Clean up
        hipMemUnmap(ptr, padded_size);
        hipMemRelease(handle);
        hipMemAddressFree(ptr, padded_size);
        return nullptr;
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
        g_allocations[ptr] = {padded_size, handle, is_host_fallback};
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
        if (g_current_device_allocated == 0 && g_current_host_allocated == 0) {
            g_fallback_active.store(false);
        }
    }

    // Cleanup strict order:
    // 1. Unmap
    // 2. Release Handle
    // 3. Free Address
    // CRITICAL: We must attempt to execute all three even if one fails to prevent catastrophic leaks.
    hipError_t status;

    status = hipMemUnmap(ptr, meta.padded_size);
    HIP_LOG_ERROR(status, "hipMemUnmap failed");

    status = hipMemRelease(meta.handle);
    HIP_LOG_ERROR(status, "hipMemRelease failed");

    status = hipMemAddressFree(ptr, meta.padded_size);
    HIP_LOG_ERROR(status, "hipMemAddressFree failed");
}

} // extern "C"
