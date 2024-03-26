#pragma once

#include <cuda_runtime_api.h>

// if in release mode, we don't want to check for cuda errors
#ifdef NDEBUG
#define CHECK_CUDA_CALL(x) x
#else
#include "cuda/common/helper_cuda.hpp"
#define CHECK_CUDA_CALL(x) checkCudaErrors(x)
#endif

template <typename T>
constexpr void malloc_managed(T** ptr, const size_t num_items)
{
	CHECK_CUDA_CALL(
		cudaMallocManaged(reinterpret_cast<void **>(ptr), num_items * sizeof(T)));
}

template <typename T>
constexpr void malloc_device(T** ptr, const size_t num_items)
{
	CHECK_CUDA_CALL(
		cudaMalloc(reinterpret_cast<void **>(ptr), num_items * sizeof(T)));
}

#define MALLOC_MANAGED(ptr, num_items) malloc_managed(ptr, num_items)

#define MALLOC_DEVICE(ptr, num_items) malloc_device(ptr, num_items)

#define SET_MEM_2_ZERO(ptr, item_count) \
  CHECK_CUDA_CALL(cudaMemsetAsync(      \
      ptr, 0, sizeof(std::remove_pointer_t<decltype(ptr)>) * (item_count)))

#define CUDA_FREE(ptr) CHECK_CUDA_CALL(cudaFree(ptr))

#define SYNC_STREAM(stream) CHECK_CUDA_CALL(cudaStreamSynchronize(stream))

#define SYNC_DEVICE() CHECK_CUDA_CALL(cudaDeviceSynchronize())
