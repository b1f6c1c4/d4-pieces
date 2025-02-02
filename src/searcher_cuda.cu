#include "searcher_cuda.h"

#include <cuda.h>
#include <cuda/atomic>
#include <cstring>
#include <iostream>
#include <format>
#include <unistd.h>
#include <cstdio>

#define MAX_SOLUTIONS (1ull << 24)

/**
 * 128 resident grids / device (Concurrent Kernel Execution)
 * 2147483647*65535*65535 blocks / grid
 * 1024*1024*64 <= 1024 threads / block
 * 32 threads / warp
 * 16 blocks / SM
 * 48 threads / warp
 * 1536 threads / SM
 * 65536 regs / SM
 * 255 regs / threads
 * 64KiB constant memory (8KiB cache)
 */

#define C(ans) { chk_impl((ans), __FILE__, __LINE__); }

static inline void chk_impl(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        throw std::runtime_error{
            std::format("CUDA: {}: {} @ {}:{}\n",
                    cudaGetErrorName(code), cudaGetErrorString(code),
                    file, line) };
    }
}

static const frow_info_t *h_frowInfoL, *h_frowInfoR;
__device__ static frow_info_t d_frowInfoL[16], d_frowInfoR[16];

void frow_cache(const frow_info_t *fiL, const frow_info_t *fiR) {
    h_frowInfoL = fiL;
    h_frowInfoR = fiR;
    for (auto i = 0; i < 16; i++) {
        frow_info_t fL{ fiL[i] }, fR{ fiR[i] };
        C(cudaMalloc(&fL.data, fiL[i].sz[5] * sizeof(frow_t)));
        C(cudaMalloc(&fR.data, fiR[i].sz[5] * sizeof(frow_t)));
        C(cudaMemcpy(fL.data, fiL[i].data, fiL[i].sz[5] * sizeof(frow_t), cudaMemcpyHostToDevice));
        C(cudaMemcpy(fR.data, fiR[i].data, fiR[i].sz[5] * sizeof(frow_t), cudaMemcpyHostToDevice));
        C(cudaMemcpyToSymbol(d_frowInfoL, &fL, sizeof(frow_info_t), i * sizeof(frow_info_t)));
        C(cudaMemcpyToSymbol(d_frowInfoR, &fR, sizeof(frow_info_t), i * sizeof(frow_info_t)));
    }
    C(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, ~0ull));
    size_t drplc;
    C(cudaDeviceGetLimit(&drplc, cudaLimitDevRuntimePendingLaunchCount));
    std::cout << std::format("DRPLC = {}\n", drplc);
}

bool h_push_nm(uint32_t &nm_cnt, uint8_t old_nm[16], frow_t &frow) {
    for (auto v = 0; v < 4; v++) {
        auto nm = frow.nm[v];
        if (nm == 0xffu)
            break;
        for (auto o = 0; o < nm_cnt; o++)
            if (old_nm[o] == nm)
                return false;
        old_nm[nm_cnt++] = nm;
    }
    return true;
}

__device__ __forceinline__
bool d_push_nm(uint32_t &nm_cnt, uint32_t old_nm[4], uint32_t new_nm) {
#pragma unroll
    for (auto v = 0; v < 4; v++) {
        auto nm = (new_nm >> 8 * v) & 0xffu;
        if (nm == 0xffu)
            break;
        auto nmx = __byte_perm(new_nm, 0, v << 0 | v << 4 | v << 8 | v << 12);
#pragma unroll
        for (auto o = 0; o < 4; o++) {
            if (4 * o >= nm_cnt)
                break;
            if (__vcmpeq4(nmx, old_nm[o])) [[unlikely]]
                return false;
        }
        __builtin_assume(nm_cnt < 16);
        old_nm[nm_cnt / 4] &= ~(0xffu << nm_cnt % 4 * 8);
        old_nm[nm_cnt / 4] |= nm << nm_cnt % 4 * 8;
        nm_cnt++;
    }
    return true;
}

void h_row_search(
        CudaSearcher::R *solutions,
        unsigned long long *n_solutions_,
        CudaSearcher::C cfg0) {
    auto szid = min(cfg0.height - 1, 5);
    auto &f0L = h_frowInfoL[cfg0.empty_area >> 0 & 0xfu];
    auto &f0R = h_frowInfoR[cfg0.empty_area >> 4 & 0xfu];
    for (auto iL = 0u; iL < f0L.sz[szid]; iL++)
        for (auto iR = 0u; iR < f0R.sz[szid]; iR++) {
            auto cfg = cfg0;
            auto &fL = f0L.data[iL];
            auto &fR = f0R.data[iR];
            if (fL.shape & ~cfg.empty_area) continue;
            if (fR.shape & ~cfg.empty_area) continue;
            if (fL.shape & fR.shape) continue;
            if (!h_push_nm(cfg.nm_cnt, reinterpret_cast<uint8_t *>(cfg.ex), fL)) continue;
            if (!h_push_nm(cfg.nm_cnt, reinterpret_cast<uint8_t *>(cfg.ex), fR)) continue;
            cfg.empty_area &= ~fL.shape;
            cfg.empty_area &= ~fR.shape;
            if (cfg.empty_area & 0b11111111u)
                throw std::runtime_error{ std::format("frow info wrong, at 0b{:08b}", cfg0.empty_area & 0xffu) };
            cfg.empty_area >>= 8;
            auto pos = cfg.empty_area & 0b11111111u;
            cuda::atomic_ref n_solutions{ n_solutions_[pos] };
            solutions[pos][n_solutions.fetch_add(1)] = cfg;
        }
}

template <bool System = false>
__global__
void d_row_search(
        CudaSearcher::R *solutions[256],
        unsigned long long *n_solutions,
        CudaSearcher::C cfg) {
    auto idx = threadIdx.x + static_cast<uint64_t>(blockIdx.x) * blockDim.x;
    auto szid = min(cfg.height - 1, 5);
    auto &f0L = d_frowInfoL[cfg.empty_area >> 0 & 0xfu];
    auto &f0R = d_frowInfoR[cfg.empty_area >> 4 & 0xfu];
    if (idx >= f0L.sz[szid] * f0R.sz[szid]) return;
    auto &fL = f0L.data[idx / f0R.sz[szid]];
    auto &fR = f0R.data[idx % f0R.sz[szid]];
    if (fL.shape & ~cfg.empty_area) return;
    if (fR.shape & ~cfg.empty_area) return;
    if (fL.shape & fR.shape) return;
    if (!d_push_nm(cfg.nm_cnt, cfg.ex, fL.nm0123)) return;
    if (!d_push_nm(cfg.nm_cnt, cfg.ex, fR.nm0123)) return;
    cfg.empty_area &= ~fL.shape;
    cfg.empty_area &= ~fR.shape;
    __builtin_assume(!(cfg.empty_area & 0b11111111u));
    cfg.empty_area >>= 8;
    auto pos = cfg.empty_area & 0b11111111u;
    CudaSearcher::R *ptr;
    if constexpr (System)
        ptr = solutions[pos] + atomicAdd_system(n_solutions + pos, 1);
    else
        ptr = solutions[pos] + atomicAdd(n_solutions + pos, 1);
    *ptr = cfg; // slice
}

CudaSearcher::CudaSearcher(uint64_t empty_area)
    : solutions{ new R[]{ empty_area, { ~0u, ~0u, ~0u, ~0u } } },
      height{ (std::bit_width(empty_area) + 8u - 1u) / 8u },
      n_solutions{ 1 },
      n_next{ next_size(0) } {
    if (status() != HOST)
        throw std::runtime_error{ "new solutions weird" };
}

CudaSearcher::~CudaSearcher() {
    free();
}

void CudaSearcher::free() {
    if (status() == HOST)
        delete [] solutions;
    else
        C(cudaFree(solutions));
    solutions = nullptr;
}

std::pair<uint64_t, uint32_t> balance(uint64_t n) {
    if (n <= 32)
        return { 1, n };
    if (n <= 32 * 84)
        return { (n + 31) / 32, 32 };
    if (n <= 64 * 84)
        return { (n + 63) / 64, 64 };
    if (n <= 96 * 84)
        return { (n + 95) / 96, 96 };
    if (n <= 128 * 84)
        return { (n + 127) / 128, 128 };
    if (n <= 256 * 84)
        return { (n + 255) / 256, 256 };
    return { (n + 511) / 512, 512 };
}

void CudaSearcher::search_GPU(mem_t mem, bool fake) {
    ensure_CPU();
    R *solutions;
    switch (mem) {
        case DEVICE:
            C(cudaMalloc(&solutions, this->n_next * sizeof(R)));
            break;
        case UNIFIED:
            C(cudaMallocManaged(&solutions, this->n_next * sizeof(R)));
            break;
        default:
            throw std::runtime_error{ "invalid mem_t" };
    }
    unsigned long long *n_solutions{}, *n_next{};
    C(cudaMalloc(&n_solutions, sizeof(unsigned long long)));
    C(cudaMalloc(&n_next, sizeof(unsigned long long)));
    for (auto i = 0zu; i < this->n_solutions; i++) {
        auto [b, t] = balance(next_size(i));
        d_row_search<<<b, t>>>(
                solutions, n_solutions, n_next, 
                C{ this->solutions[i], height });
        C(cudaPeekAtLastError());
    }
    C(cudaDeviceSynchronize());
    if (!fake) {
        C(cudaMemcpy(&this->n_solutions, n_solutions, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        C(cudaMemcpy(&this->n_next, n_next, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        free();
        this->solutions = solutions;
        height--;
    } else {
        C(cudaFree(solutions));
    }
    C(cudaFree(n_solutions));
    C(cudaFree(n_next));
}

void CudaSearcher::search_Mixed(uint64_t threshold, bool fake) {
    ensure_CPU();
    R *solutions;
    C(cudaMallocManaged(&solutions, this->n_next * sizeof(R)));
    unsigned long long h_n_solutions{}, h_n_next{};
    unsigned long long *d_n_solutions{}, *d_n_next{};
    C(cudaMallocManaged(&d_n_solutions, sizeof(unsigned long long)));
    C(cudaMallocManaged(&d_n_next, sizeof(unsigned long long)));
    for (auto i = 0zu; i < this->n_solutions; i++) {
        auto ns = next_size(i);
        auto [b, t] = balance(ns);
        if (ns < threshold)
            h_row_search<-1ll>(
                    solutions + this->n_next - 1, &h_n_solutions, &h_n_next, 
                    C{ this->solutions[i], height });
        else 
            d_row_search<<<b, t>>>(
                    solutions, d_n_solutions, d_n_next, 
                    C{ this->solutions[i], height });
        C(cudaPeekAtLastError());
    }
    C(cudaDeviceSynchronize());
    if (!fake) {
        std::memmove(
                solutions + *d_n_solutions,
                solutions + this->n_next - h_n_solutions,
                h_n_solutions * sizeof(unsigned long long));
        this->n_solutions = h_n_solutions + *d_n_solutions;
        this->n_next = h_n_next + *d_n_next;
        free();
        this->solutions = solutions;
        height--;
    } else {
        C(cudaFree(solutions));
    }
    C(cudaFree(d_n_solutions));
    C(cudaFree(d_n_next));
}

uint64_t CudaSearcher::next_size(uint64_t i) const {
    auto ea = solutions[i].empty_area;
    auto nszid = min(height - 2, 5);
    return
        h_frowInfoL[ea >> 0 & 0xfu].sz[nszid] *
        h_frowInfoR[ea >> 4 & 0xfu].sz[nszid];
}

void CudaSearcher::ensure_CPU() {
    switch (status()) {
        case HOST:
        case UNIFIED:
            return;
        case DEVICE:
            break;
        default:
            throw std::runtime_error{ "invalid mem_t" };
    }
    auto solutions = new R[n_solutions];
    C(cudaMemcpy(solutions, this->solutions, n_solutions * sizeof(R), cudaMemcpyDeviceToHost));
    C(cudaFree(this->solutions));
    this->solutions = solutions;
}

CudaSearcher::mem_t CudaSearcher::status() const {
    if (!solutions)
        return EMPTY;
    int ret;
    auto res = cuPointerGetAttribute(&ret,
                CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                (CUdeviceptr)solutions);
    switch (res) {
        case CUDA_SUCCESS:
            return static_cast<mem_t>(ret);
        case CUDA_ERROR_DEINITIALIZED:
            throw std::runtime_error{ "cuPointerGetAttribute failed: deinitialized" };
        case CUDA_ERROR_NOT_INITIALIZED:
            throw std::runtime_error{ "cuPointerGetAttribute failed: not initialized" };
        case CUDA_ERROR_INVALID_CONTEXT:
            throw std::runtime_error{ "cuPointerGetAttribute failed: invalid context" };
        case CUDA_ERROR_INVALID_VALUE:
            return HOST;
            // throw std::runtime_error{ "cuPointerGetAttribute failed: invalid value" };
        case CUDA_ERROR_INVALID_DEVICE:
            throw std::runtime_error{ "cuPointerGetAttribute failed: invalid device" };
        default:
            throw std::runtime_error{ "cuPointerGetAttribute failed" };
    }
}

void show_devices() {
  int nDevices;
  C(cudaGetDeviceCount(&nDevices));

  printf("Number of devices: %d\n", nDevices);

  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    C(cudaGetDeviceProperties(&prop, i));
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Capability: %d.%d\n", prop.major, prop.minor);
    printf("  MP: %d\n", prop.multiProcessorCount);
    printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate/1024);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.1f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Warp-size: %d\n", prop.warpSize);
    printf("  Threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("  Threads/mp: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Async engines: %d\n", prop.asyncEngineCount);
    printf("  32-bit Registers per block: %d\n", prop.regsPerBlock);
    printf("  32-bit Registers per mp: %d\n", prop.regsPerMultiprocessor);
    printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Concurrent computation/communication: %s\n",prop.deviceOverlap ? "yes" : "no");
    printf("  Sparse: %s\n",prop.sparseCudaArraySupported ? "yes" : "no");
    printf("  Managed mem: %s\n",prop.managedMemory ? "yes" : "no");
    printf("  Deferred: %s\n",prop.deferredMappingCudaArraySupported ? "yes" : "no");
    printf("  Map host mem: %s\n",prop.canMapHostMemory ? "yes" : "no");
    printf("  Unified addr: %s\n",prop.unifiedAddressing ? "yes" : "no");
    printf("  Unified fp: %s\n",prop.unifiedFunctionPointers ? "yes" : "no");
    printf("  Concurrent managed access: %s\n",prop.concurrentManagedAccess ? "yes" : "no");
    printf("  PMA: %s\n",prop.pageableMemoryAccess ? "yes" : "no");
    printf("  ECC: %s\n",prop.ECCEnabled ? "yes" : "no");
    printf("  Cooperative launch: %s\n",prop.cooperativeLaunch ? "yes" : "no");
    printf("  DMMA from host: %s\n",prop.directManagedMemAccessFromHost ? "yes" : "no");
    printf("  L2 Cache Size (KiB): %d\n",prop.l2CacheSize / 1024);
    printf("  Shared mem per block (KiB): %lu\n",prop.sharedMemPerBlock / 1024);
    printf("  Shared mem per mp (KiB): %lu\n",prop.sharedMemPerMultiprocessor / 1024);
    printf("  Const mem (B): %lu\n",prop.totalConstMem / 1024);
    printf("  Global mem (MiB): %lf\n",prop.totalGlobalMem / 1024.0 / 1024);
    int v;
    cudaDeviceGetAttribute(&v, cudaDevAttrMemSyncDomainCount, i);
    printf("  Sync domain: %d\n",v);
    cudaDeviceGetAttribute(&v, cudaDevAttrSingleToDoublePrecisionPerfRatio, i);
    printf("  float/double ratio: %d\n", v);
    cudaDeviceGetAttribute(&v, (cudaDeviceAttr)102, i);
    printf("  VMM: %d\n", v);
  }
}
