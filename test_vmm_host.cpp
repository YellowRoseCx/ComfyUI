#include <hip/hip_runtime.h>
#include <iostream>

#define ROUND_UP(SIZE,GRANULARITY) ((1 + SIZE / GRANULARITY) * GRANULARITY)

#define HIP_CHECK(expression)              \
{                                          \
    const hipError_t err = expression;     \
    if(err != hipSuccess){                 \
        std::cerr << "HIP error: "         \
            << hipGetErrorString(err)      \
            << " at " << __LINE__ << "\n"; \
    }                                      \
}

__global__ void zeroAddr(int* pointer) {
    *pointer = 0;
}

__global__ void fillAddr(int* pointer) {
    *pointer = 42;
}

int main() {
    int currentDev = 0;
    hipDeviceProp_t propDev;
    hipGetDeviceProperties(&propDev, currentDev);
    std::cout << "Executing on: " << propDev.name << "\n" << std::endl;

    int vmm = 0;
    HIP_CHECK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, currentDev));

    std::cout << "Virtual memory management support value: " << vmm << std::endl;

    if (vmm == 0) {
        std::cout << "GPU 0 doesn't support virtual memory management.";
        return 0;
    }

    size_t size = 4 * 1024;
    hipMemGenericAllocationHandle_t allocHandle;
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;

    // **********************************************
    // CRITICAL CHANGE: Testing Host Memory Fallback
    // **********************************************
    prop.location.type = hipMemLocationTypeHost;
    prop.location.id = 0;

    size_t granularity = 0;
    HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));

    size_t padded_size = ROUND_UP(size, granularity);
    HIP_CHECK(hipMemCreate(&allocHandle, padded_size * 2, &prop, 0));

    void* virtualPointer = nullptr;
    HIP_CHECK(hipMemAddressReserve(&virtualPointer, padded_size, granularity, nullptr, 0));

    HIP_CHECK(hipMemMap(virtualPointer, padded_size, 0, allocHandle, 0));

    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = currentDev;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;

    HIP_CHECK(hipMemSetAccess(virtualPointer, padded_size, &accessDesc, 1));

    int value = 42;
    HIP_CHECK(hipMemcpy(virtualPointer, &value, sizeof(int), hipMemcpyHostToDevice));

    int result = 1;
    HIP_CHECK(hipMemcpy(&result, virtualPointer, sizeof(int), hipMemcpyDeviceToHost));
    if( result == 42) {
        std::cout << "Success. Value: " << result << std::endl;
    } else {
        std::cout << "Failure. Value: " << result << std::endl;
    }

    std::cout << "Launching zeroAddr kernel to test if AMD GPU compute can read Host VMM memory..." << std::endl;

    // Launch zeroAddr kernel
    zeroAddr<<<1, 1>>>((int*)virtualPointer);
    hipError_t kernel_err = hipDeviceSynchronize();

    if (kernel_err != hipSuccess) {
        std::cerr << "KERNEL FAULT: " << hipGetErrorString(kernel_err) << "\n" << std::endl;
        std::cerr << "CONCLUSION: The RX 6800 XT physically faults when executing compute shaders on hipMemLocationTypeHost VMM handles." << std::endl;
        return 1;
    }

    result = 1;
    HIP_CHECK(hipMemcpy(&result, virtualPointer, sizeof(int), hipMemcpyDeviceToHost));
    if( result == 0) {
        std::cout << "Success. zeroAddr kernel: " << result << std::endl;
    } else {
        std::cout << "Failure. zeroAddr kernel: " << result << std::endl;
    }

    fillAddr<<<1, 1>>>((int*)virtualPointer);
    HIP_CHECK(hipDeviceSynchronize());

    result = 1;
    HIP_CHECK(hipMemcpy(&result, virtualPointer, sizeof(int), hipMemcpyDeviceToHost));
    if( result == 42) {
        std::cout << "Success. fillAddr kernel: " << result << std::endl;
    } else {
        std::cout << "Failure. fillAddr kernel: " << result << std::endl;
    }

    HIP_CHECK(hipMemUnmap(virtualPointer, padded_size));
    HIP_CHECK(hipMemRelease(allocHandle));
    HIP_CHECK(hipMemAddressFree(virtualPointer, padded_size));

    return 0;
}