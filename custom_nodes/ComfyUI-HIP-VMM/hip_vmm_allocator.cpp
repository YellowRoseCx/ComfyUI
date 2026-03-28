#include <hip/hip_runtime.h>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <vector>
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
struct AllocationChunk {
    hipMemGenericAllocationHandle_t handle;
    size_t offset;
    size_t size;
};

struct AllocationMeta {
    size_t padded_size;
    std::vector<AllocationChunk> chunks;
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

    std::vector<AllocationChunk> chunks;
    bool is_host_fallback = false;

    // 4. Attempt Device Allocation
    if (g_vram_limit_bytes == 0 || (g_current_device_allocated + padded_size <= g_vram_limit_bytes)) {
        hipMemGenericAllocationHandle_t dev_handle;
        // AMD ROCm Official Docs: hipMemCreate physical handle must be padded_size * 2
        // to provide adequate internal page alignment slack, preventing unspecified launch failures
        status = hipMemCreate(&dev_handle, padded_size * 2, &device_prop, 0);
        if (status == hipSuccess) {
            chunks.push_back({dev_handle, 0, padded_size});
        }
    }

    // 5. The Fallback (System RAM mapped to GPU Virtual Address via Slicing)
    if (chunks.empty()) {
        if (!g_fallback_active.exchange(true)) {
             std::cout << "[HIP VMM WARN] VRAM limit reached or allocation failed natively, falling back to Fragmented Host RAM over PCIe" << std::endl;
        }

        // Verify Host allocation doesn't exceed user RAM limit
        if (g_ram_limit_bytes > 0 && (g_current_host_allocated + padded_size > g_ram_limit_bytes)) {
            std::cerr << "[HIP VMM ERROR] Host RAM limit exceeded! Attempted to allocate " << (padded_size / (1024.0 * 1024.0))
                      << " MB (Currently allocated: " << (g_current_host_allocated / (1024.0 * 1024.0)) << " MB)" << std::endl;
            hipMemAddressFree(ptr, padded_size);
            return nullptr;
        }

        is_host_fallback = true;

        // VMM SLICING: Linux physical memory fragmentation prevents hipMemCreate from
        // allocating large contiguous pages (e.g., 3GB) in System RAM. If we ask for 3GB,
        // the kernel rejects it with `hipErrorOutOfMemory` even if 30GB is collectively free.
        // We slice the request into smaller chunks and map them seamlessly.

        // 256MB chunk size (must be a multiple of max_granularity)
        size_t slice_size = std::max((size_t)(256 * 1024 * 1024), max_granularity);
        slice_size = ((slice_size + max_granularity - 1) / max_granularity) * max_granularity;

        size_t allocated_so_far = 0;
        bool fallback_failed = false;

        while (allocated_so_far < padded_size) {
            size_t current_slice = std::min(slice_size, padded_size - allocated_so_far);
            hipMemGenericAllocationHandle_t host_handle;

            // AMD ROCm Official Docs: physical handle must be slice * 2
            status = hipMemCreate(&host_handle, current_slice * 2, &host_prop, 0);

            // If even 256MB fails, iteratively halve the slice size down to the max_granularity limit
            while (status != hipSuccess && current_slice > max_granularity) {
                current_slice = std::max(current_slice / 2, max_granularity);
                current_slice = ((current_slice + max_granularity - 1) / max_granularity) * max_granularity;
                status = hipMemCreate(&host_handle, current_slice * 2, &host_prop, 0);
            }

            if (status != hipSuccess) {
                std::cerr << "\n========================================\n";
                std::cerr << "[HIP VMM FATAL] Host fragmented memory allocation failed: " << hipGetErrorString(status) << "\n";
                std::cerr << "[HIP VMM DIAGNOSTIC] Failed at chunk " << chunks.size() << " (" << (current_slice / (1024.0 * 1024.0)) << " MB).\n";
                std::cerr << "[HIP VMM DIAGNOSTIC] Current VMM Host Tracker: " << (g_current_host_allocated / (1024.0 * 1024.0)) << " MB.\n";
                std::cerr << "--> WHY DID THIS CRASH?\n";
                std::cerr << "Linux limits the amount of 'pinned' (page-locked) RAM a user can request to prevent freezing the OS.\n";
                std::cerr << "Your current limit ('ulimit -l') is too low to absorb this PyTorch fallback allocation request.\n";
                std::cerr << "SOLUTION: Run 'ulimit -l unlimited' (or edit /etc/security/limits.conf) before launching ComfyUI.\n";
                std::cerr << "========================================\n" << std::endl;
                fallback_failed = true;
                break;
            }

            chunks.push_back({host_handle, allocated_so_far, current_slice});
            allocated_so_far += current_slice;
        }

        if (fallback_failed) {
            for (const auto& chunk : chunks) {
                hipMemRelease(chunk.handle);
            }
            hipMemAddressFree(ptr, padded_size);
            return nullptr;
        }

    } else {
        // Device allocation succeeded, we are not in fallback for this allocation
        if (g_fallback_active.load() && g_current_device_allocated == 0) {
            // Optional: reset fallback log when memory drops completely
            g_fallback_active.store(false);
        }
    }

    // 6. Map Physical Memory Chunks to Virtual Address dynamically
    for (const auto& chunk : chunks) {
        // Compute absolute offset mapped pointer
        void* mapped_ptr = static_cast<char*>(ptr) + chunk.offset;
        status = hipMemMap(mapped_ptr, chunk.size, 0, chunk.handle, 0);

        if (status != hipSuccess) {
            std::cerr << "[HIP VMM ERROR] hipMemMap failed on slice size " << chunk.size << ": " << hipGetErrorString(status) << std::endl;
            for (const auto& c : chunks) {
                hipMemRelease(c.handle); // Just release handles; we cannot partially unmap what wasn't successfully mapped
            }
            hipMemAddressFree(ptr, padded_size);
            return nullptr;
        }
    }

    // 7. Grant Access to the entire contiguous virtual block
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = device;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;

    status = hipMemSetAccess(ptr, padded_size, &accessDesc, 1);
    if (status != hipSuccess) {
        std::cerr << "[HIP VMM ERROR] hipMemSetAccess failed: " << hipGetErrorString(status) << std::endl;

        // Clean up
        for (const auto& chunk : chunks) {
            void* mapped_ptr = static_cast<char*>(ptr) + chunk.offset;
            hipMemUnmap(mapped_ptr, chunk.size);
            hipMemRelease(chunk.handle);
        }
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
        g_allocations[ptr] = {padded_size, chunks, is_host_fallback};
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
    // 1. Unmap individual chunks
    // 2. Release Handle individual chunks
    // 3. Free contiguous virtual Address
    // CRITICAL: We must attempt to execute all even if one fails to prevent catastrophic leaks.

    hipError_t status;

    for (const auto& chunk : meta.chunks) {
        void* mapped_ptr = static_cast<char*>(ptr) + chunk.offset;
        status = hipMemUnmap(mapped_ptr, chunk.size);
        HIP_LOG_ERROR(status, "hipMemUnmap failed on slice");

        status = hipMemRelease(chunk.handle);
        HIP_LOG_ERROR(status, "hipMemRelease failed on slice");
    }

    status = hipMemAddressFree(ptr, meta.padded_size);
    HIP_LOG_ERROR(status, "hipMemAddressFree failed");
}

} // extern "C"
