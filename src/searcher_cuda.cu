#include "searcher_cuda.h"
#include "growable.cuh"

#include <cuda.h>
#include <cuda/atomic>
#include <cstring>
#include <iostream>
#include <format>
#include <unistd.h>
#include <cstdio>

#define MAX_SOLUTIONS (1ull << 24)

inline bool operator<(const CudaSearcher::R &lhs, const CudaSearcher::R &rhs) {
    if (lhs.empty_area < rhs.empty_area)
        return true;
    if (lhs.empty_area > rhs.empty_area)
        return false;
    if (lhs.nm_cnt < rhs.nm_cnt)
        return true;
    if (lhs.nm_cnt > rhs.nm_cnt)
        return false;
    for (auto o = 0; o < 4; o++) {
        if (lhs.ex[o] < rhs.ex[o])
            return false;
        if (lhs.ex[o] > rhs.ex[o])
            return false;
    }
    return false;
}

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

void chk_impl(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        throw std::runtime_error{
            std::format("CUDA: {}: {} @ {}:{}\n",
                    cudaGetErrorName(code), cudaGetErrorString(code),
                    file, line) };
    }
}

void chk_impl(CUresult code, const char *file, int line) {
    const char *pn = "???", *ps = "???";
    cuGetErrorName(code, &pn);
    cuGetErrorString(code, &ps);
    if (code != CUDA_SUCCESS) {
        throw std::runtime_error{
            std::format("CUDA Driver: {}: {} @ {}:{}\n", pn, ps, file, line) };
    }
}

static const frow_info_t *h_frowInfoL, *h_frowInfoR;
static frow_t *d_frowDataL[16], *d_frowDataR[16];

void frow_cache(const frow_info_t *fiL, const frow_info_t *fiR) {
    h_frowInfoL = fiL;
    h_frowInfoR = fiR;
    for (auto i = 0; i < 16; i++) {
        C(cudaMalloc(&d_frowDataL[i], fiL[i].sz[5] * sizeof(frow_t)));
        C(cudaMalloc(&d_frowDataR[i], fiR[i].sz[5] * sizeof(frow_t)));
        C(cudaMemcpy(d_frowDataL[i], fiL[i].data, fiL[i].sz[5] * sizeof(frow_t), cudaMemcpyHostToDevice));
        C(cudaMemcpy(d_frowDataR[i], fiR[i].data, fiR[i].sz[5] * sizeof(frow_t), cudaMemcpyHostToDevice));
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

template <bool System = false>
__global__
void d_row_search(CudaSearcher::B bins,
        const CudaSearcher::R *cfgs, uint64_t n_cfgs,
        const frow_t *f0L, uint32_t f0Lsz,
        const frow_t *f0R, uint32_t f0Rsz) {
    auto idx = threadIdx.x + static_cast<uint64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_cfgs * f0Lsz * f0Rsz) return;
    auto cfg = cfgs[idx / f0Rsz / f0Lsz];
    auto fL  = f0L [idx / f0Rsz % f0Lsz];
    auto fR  = f0R [idx % f0Rsz];
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
        ptr = bins[pos].ptr + atomicAdd_system(&bins[pos].len, 1);
    else
        ptr = bins[pos].ptr + atomicAdd(&bins[pos].len, 1);
    *ptr = cfg; // slice
}

CudaSearcher::CudaSearcher(uint64_t empty_area)
    : solutions{}, height{ (std::bit_width(empty_area) + 8u - 1u) / 8u } {
    auto &r = solutions[empty_area & 0xffu];
    C(cudaMallocManaged(&r.ptr, sizeof(R)));
    r.ptr[0] = R{ empty_area, { ~0u, ~0u, ~0u, ~0u }, 0 };
    r.len = 1;
}

CudaSearcher::~CudaSearcher() {
    free();
}

void CudaSearcher::free() {
    for (auto &r : solutions) {
        if (r.ptr)
            cudaFree(r.ptr);
        r.ptr = nullptr;
        r.len = 0;
    }
}

std::pair<uint64_t, uint32_t> balance(uint64_t n) {
    if (n <= 32)
        return { 1, n };
    if (n <= 32 * 84 * 3)
        return { (n + 31) / 32, 32 };
    if (n <= 64 * 84 * 3)
        return { (n + 63) / 64, 64 };
    if (n <= 96 * 84 * 3)
        return { (n + 95) / 96, 96 };
    if (n <= 128 * 84 * 3)
        return { (n + 127) / 128, 128 };
    if (n <= 256 * 84 * 3)
        return { (n + 255) / 256, 256 };
    return { (n + 511) / 512, 512 };
}

void CudaSearcher::search_GPU(bool fake) {
    Growable<R> grs[256];
    Rg<R> *bins;
    C(cudaMallocManaged(&bins, sizeof(B)));
    C(cudaMemset(bins, 0, sizeof(B)));
    for (auto ipos = 0u; ipos <= 255u; ipos++) {
        auto [ptr, len] = solutions[ipos];
        auto &f0L = h_frowInfoL[ipos >> 0 & 0xfu];
        auto &f0R = h_frowInfoR[ipos >> 4 & 0xfu];
        auto d_f0L = d_frowDataL[ipos >> 0 & 0xfu];
        auto d_f0R = d_frowDataR[ipos >> 4 & 0xfu];
        auto szid = min(height - 1, 5);
        auto fanout = (unsigned long long)f0L.sz[szid] * f0R.sz[szid];
        auto risk_free = len * fanout;
        for (auto opos = 0u; opos <= 255u; opos++)
            risk_free = min(risk_free, (unsigned long long)grs[opos].risk_free_size());
        auto alloc = len * fanout;
        alloc = max(risk_free, min(alloc, 16ull * 1048576 / sizeof(R)));
        auto max_n = max(1ull, alloc / fanout);
        if (len / max_n > 100)
            std::cout << std::format("ipos=0b{:08b} len={} max_n={} fanout={} len*fanout*sizeof(R)={}\n",
                    ipos, len, max_n, fanout, display(len*fanout*sizeof(R)));
        while (len) {
            auto n = min(len, max_n);
            auto [b, t] = balance(n * fanout);
            for (auto opos = 0u; opos <= 255u; opos++) {
                bins[opos].ptr = grs[opos].get(alloc);
                bins[opos].len = 0;
                if (!bins[opos].ptr)
                    throw std::runtime_error{ "OOM" };
            }
            d_row_search<<<b, t>>>(bins, ptr, n,
                    d_f0L, f0L.sz[szid],
                    d_f0R, f0R.sz[szid]);
            C(cudaPeekAtLastError());
            C(cudaDeviceSynchronize());
            for (auto opos = 0u; opos <= 255u; opos++)
                grs[opos].commit(bins[opos].len);
            len -= n;
        }
    }
    if (!fake) {
        for (auto ipos = 0u; ipos <= 255u; ipos++) {
            C(cudaFree(solutions[ipos].ptr));
            solutions[ipos] = grs[ipos].cpu_merge_sort();
        }
        height--;
    } else {
        for (auto ipos = 0u; ipos <= 255u; ipos++) {
            auto [ptr, len] = grs[ipos].cpu_merge_sort();
            C(cudaFree(ptr));
        }
    }
    C(cudaFree(bins));
}

uint64_t CudaSearcher::next_size() const {
    auto szid = min(height - 1, 5);
    auto cnt = 0ull;
    for (auto pos = 0u; pos <= 255u; pos++)
        cnt += solutions[pos].len
            * h_frowInfoL[(pos >> 0) & 0b1111u].sz[szid]
            * h_frowInfoR[(pos >> 4) & 0b1111u].sz[szid];
    return cnt;
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
