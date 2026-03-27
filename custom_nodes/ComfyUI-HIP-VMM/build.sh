#!/bin/bash
set -e

# Build the PyTorch HIP VMM Pluggable Allocator for AMD gfx1030 (RX 6800 XT)

HIPCC_FLAGS="-O3 -fPIC -shared --amdgpu-target=gfx1030 -std=c++17"

echo "Compiling HIP VMM Allocator for gfx1030..."
hipcc $HIPCC_FLAGS -o hip_vmm_allocator.so hip_vmm_allocator.cpp

echo "Build complete. Shared library 'hip_vmm_allocator.so' generated."
