#include <hip/hip_runtime.h>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <algorithm> // for std::max

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

    static std::atomic<bool> g_first_alloc(true);
    if (g_first_alloc.exchange(false)) {
        std::cout << "[HIP VMM] Verified: Custom my_malloc intercept successfully reached." << std::endl;
    }

    // MULTI-GPU SAFETY: Always set the executing device to prevent VMM structures
    // from attaching to the wrong GPU's page tables and causing unspecified launch failures.
    hipError_t status = hipSetDevice(device);
    if (status != hipSuccess) {
        std::cerr << "[HIP VMM FATAL] Failed to set device context: " << hipGetErrorString(status) << std::endl;
        return nullptr;
    }

    // 1. Get Allocation Granularities for Device and Host
    hipMemAllocationProp device_prop = {};
    device_prop.type = hipMemAllocationTypePinned;
    device_prop.location.type = hipMemLocationTypeDevice;
    device_prop.location.id = device;

    size_t device_granularity = 0;
    status = hipMemGetAllocationGranularity(&device_granularity, &device_prop, hipMemAllocationGranularityMinimum);
    if (status != hipSuccess) {
        std::cerr << "[HIP VMM ERROR] Device Granularity failed: " << hipGetErrorString(status) << std::endl;
        return nullptr;
    }

    hipMemAllocationProp host_prop = {};
    host_prop.type = hipMemAllocationTypePinned;
    host_prop.location.type = hipMemLocationTypeHost;
    host_prop.location.id = 0; // Host ID is ignored by spec

    size_t host_granularity = 0;
    status = hipMemGetAllocationGranularity(&host_granularity, &host_prop, hipMemAllocationGranularityMinimum);
    if (status != hipSuccess) {
        std::cerr << "[HIP VMM ERROR] Host Granularity failed: " << hipGetErrorString(status) << std::endl;
        return nullptr;
    }

    // Use the largest granularity required so the Virtual Address space can dynamically
    // satisfy either physical backing natively without hipErrorOutOfMemory alignment rejections
    size_t max_granularity = std::max(device_granularity, host_granularity);

    // 2. Pad size mathematically perfectly
    size_t padded_size = ((size + max_granularity - 1) / max_granularity) * max_granularity;

    // 3. Reserve Virtual Address Space with EXPLICIT max_granularity alignment
    void* ptr = nullptr;
    HIP_CHECK(hipMemAddressReserve(&ptr, padded_size, max_granularity, nullptr, 0));

    hipMemGenericAllocationHandle_t handle;
    bool is_host_fallback = false;

    // 4. Attempt Device Allocation
    if (g_vram_limit_bytes > 0 && (g_current_device_allocated + padded_size > g_vram_limit_bytes)) {
        // Exceeds VRAM limit natively, skip directly to fallback
        status = hipErrorOutOfMemory;
    } else {
        // AMD ROCm Official Docs: hipMemCreate physical handle must be padded_size * 2
        // to provide adequate internal page alignment slack, preventing unspecified launch failures
        status = hipMemCreate(&handle, padded_size * 2, &device_prop, 0);
    }

    // 5. The Fallback (System RAM mapped to GPU Virtual Address)
    if (status != hipSuccess) {
        if (!g_fallback_active.exchange(true)) {
             std::cout << "[HIP VMM WARN] VRAM limit reached or Device allocation failed, falling back to Pinned Host RAM over PCIe" << std::endl;
        }

        // Verify Host allocation doesn't exceed user RAM limit
        if (g_ram_limit_bytes > 0 && (g_current_host_allocated + padded_size > g_ram_limit_bytes)) {
            std::cerr << "[HIP VMM ERROR] Host RAM limit exceeded! Attempted to allocate " << (padded_size / (1024.0 * 1024.0))
                      << " MB (Currently allocated: " << (g_current_host_allocated / (1024.0 * 1024.0)) << " MB)" << std::endl;
            hipMemAddressFree(ptr, padded_size);
            return nullptr;
        }

        // Create the physical memory in Pinned System RAM using the precise Host VMM properties
        // AMD ROCm Official Docs: physical handle must be padded_size * 2
        status = hipMemCreate(&handle, padded_size * 2, &host_prop, 0);
        if (status != hipSuccess) {
            std::cerr << "\n========================================\n";
            std::cerr << "[HIP VMM FATAL] Host memory allocation failed: " << hipGetErrorString(status) << "\n";
            std::cerr << "[HIP VMM DIAGNOSTIC] Attempted to lock a contiguous " << (padded_size / (1024.0 * 1024.0)) << " MB block in System RAM.\n";
            std::cerr << "[HIP VMM DIAGNOSTIC] Current VMM Host Tracker: " << (g_current_host_allocated / (1024.0 * 1024.0)) << " MB.\n";
            std::cerr << "--> WHY DID THIS CRASH?\n";
            std::cerr << "Linux OS memory fragmentation prevents hipMemCreate from finding a perfect, physically unbroken block of this size in your DDR4.\n";
            std::cerr << "Even if you have 30GB+ of RAM free, an OS that has been running for hours rarely has a perfectly contiguous multi-gigabyte segment remaining.\n";
            std::cerr << "We cannot slice this allocation to bypass fragmentation because PyTorch Triton/MIOPEN kernels fatally crash on RDNA2 when accessing virtual addresses backed by disjointed physical handles.\n";
            std::cerr << "SOLUTION: You must rely on PyTorch expandable_segments:True or native comfy_aimdo VRAM offloading to survive this specific payload.\n";
            std::cerr << "========================================\n" << std::endl;

            hipMemAddressFree(ptr, padded_size);
            return nullptr;
        }

        is_host_fallback = true;
    } else {
        // Device allocation succeeded, we are not in fallback for this allocation
        if (g_fallback_active.load() && g_current_device_allocated == 0) {
            // Optional: reset fallback log when memory drops completely
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

    // 7. Grant Access to the executing GPU
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

    // MULTI-GPU SAFETY: Set the executing device for proper page table unmapping
    hipSetDevice(device);

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

    // VMM Cleanup strict order:
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