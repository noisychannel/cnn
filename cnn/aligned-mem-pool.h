#ifndef CNN_ALIGNED_MEM_POOL_H
#define CNN_ALIGNED_MEM_POOL_H

#include <cstdlib>
#include <cstring>
#include <iostream>
//#if HAVE_MM_MALLOC
#include <mm_malloc.h>
//#endif
#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace cnn {

inline void* cnn_mm_malloc(size_t n, size_t align) {
//#if HAVE_MM_MALLOC
  void* ptr = nullptr;
#if HAVE_CUDA
  if (cudaMalloc(&ptr, n) != cudaSuccess) {
    ptr = nullptr;
  } else {
    std::cerr << "cudaMalloc succeeded: ptr=" << ptr << std::endl;
  }
#else
  ptr = _mm_malloc(n, align);
#endif
//#else
//  return std::malloc(n, align);
//#endif
  if (!ptr) {
    std::cerr << "Memory allocation failed n=" << n << " align=" << align << std::endl;
    abort();
  }
  return ptr;
}

inline void cnn_mm_free(void* mem) {
//#if HAVE_MM_MALLOC
#if HAVE_CUDA
  if (cudaFree(mem)) {
    std::cerr << "cudaFree failed\n";
    abort();
  }
#else
  _mm_free(mem);
#endif

//#else
//  return std::free(n, align);
//#endif
}

// this is used to manage CPU memory for function values and gradients
template <unsigned AlignedBits>
class AlignedMemoryPool {
 public:
  explicit AlignedMemoryPool(size_t cap) {
    sys_alloc(cap);
    zero();
  }
  // returns nullptr if OOM
  void* allocate(size_t n) {
    auto rounded_n = round_up_align(n);
    if (rounded_n + used > capacity)
      return nullptr;
    void* res = static_cast<char*>(mem) + used;
    used += rounded_n;
    return res;
  }
  void free() {
    //std::cerr << "freeing " << used << " bytes\n";
    used = 0;
  }
  void zero_and_free() { zero(); free(); }
  void free_and_grow_capacity(size_t new_cap = 0) {
    cnn_mm_free(mem);
    if (new_cap)
      sys_alloc(new_cap);
    else
      sys_alloc(capacity * 1.5);
    zero();
  }
 private:
  void sys_alloc(size_t cap) {
    capacity = round_up_align(cap);
    mem = cnn_mm_malloc(capacity, 1 << AlignedBits);
    used = 0;
  }
  void zero() {
    //std::cerr << "zeroing " << (used ? used : capacity) << " bytes\n";
#if HAVE_CUDA
    cudaMemset(mem, 0, used ? used : capacity);
#else
    std::memset(mem, 0, used ? used : capacity);
#endif
  }
  inline static size_t round_up_align(size_t n) {
    if (AlignedBits < 2) return n;
    auto c = (n & ((1 << AlignedBits) - 1)) > 0 ? 1 : 0;
    return ((n >> AlignedBits) + c) << AlignedBits;
  }
  size_t capacity;
  size_t used;
  void* mem;
};

} // namespace cnn

#endif
