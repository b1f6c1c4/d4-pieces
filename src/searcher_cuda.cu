#include "searcher_cuda.h"

#include <cuda/atomic>

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
        std::cerr << std::format("CUDA: {} @ {}:{}\n", cudaGetErrorString(code), file, line);
    }
}

__host__ static const frow_info_t *h_frowInfoL, *h_frowInfoR;
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
        C(cudaMemcpyToSymbol(d_frowInfoL, fiL, sizeof(frow_info_t), i * sizeof(frow_info_t)));
        C(cudaMemcpyToSymbol(d_frowInfoR, fiR, sizeof(frow_info_t), i * sizeof(frow_info_t)));
    }
    C(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, ~0ull));
    size_t drplc;
    C(cudaDeviceGetLimit(&drplc, cudaLimitDevRuntimePendingLaunchCount));
    std::cout << std::format("DRPLC = {}\n", drplc);
}

bool h_push_nm(uint32_t &nm_cnt, uint8_t old_nm[16], frow_t &frow) {
    __builtin_assume(nm_cnt < 16);
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

void h_row_search(uint64_t idx,
        CudaSearcher::R *solutions,
        uint64_t *n_solutions_,
        uint64_t *n_next_,
        CudaSearcher::C cfg) {
    std::atomic_ref n_solutions{ n_solutions_ };
    std::atomic_ref n_next{ n_next_ };
    auto szid = min(cfg.height - 1, 5);
    auto &f0L = h_frowInfoL[cfg.empty_area >> 0 & 0xfu];
    auto &f0R = h_frowInfoR[cfg.empty_area >> 4 & 0xfu];
    auto &fL = f0L.data[idx / f0R.sz[szid]];
    auto &fR = f0R.data[idx % f0R.sz[szid]];
    if (fL.shape & ~cfg.empty_area) return;
    if (fR.shape & ~cfg.empty_area) return;
    if (fL.shape & fR.shape) return;
    if (!h_push_nm(cfg.nm_cnt, reinterpret_cast<uint8_t *>(cfg.ex), fL)) return;
    if (!h_push_nm(cfg.nm_cnt, reinterpret_cast<uint8_t *>(cfg.ex), fR)) return;
    cfg.empty_area &= ~fL.shape;
    cfg.empty_area &= ~fR.shape;
    solutions[n_solutions++] = cfg; // slice
    if (cfg.height > 1) {
        auto nszid = min(cfg.height - 2, 5);
        auto nsz = d_frowInfoL[cfg.empty_area >> 0 & 0xfu].sz[nszid] *
            d_frowInfoR[cfg.empty_area >> 0 & 0xfu].sz[nszid];
        n_next += nsz;
    }
}

__global__
void d_row_search(
        CudaSearcher::R *solutions,
        uint64_t *n_solutions,
        uint64_t *n_next,
        CudaSearcher::C cfg) {
     auto idx = threadIdx.x + static_cast<uint64_t>(blockIdx.x) * blockDim.x;
     auto szid = min(cfg.height - 1, 5);
     auto &f0L = d_frowInfoL[cfg.empty_area >> 0 & 0xfu];
     auto &f0R = d_frowInfoR[cfg.empty_area >> 4 & 0xfu];
     auto &fL = f0L.data[idx / f0R.sz[szid]];
     auto &fR = f0R.data[idx % f0R.sz[szid]];
     if (fL.shape & ~cfg.empty_area) return;
     if (fR.shape & ~cfg.empty_area) return;
     if (fL.shape & fR.shape) return;
     if (!d_push_nm(cfg.nm_cnt, cfg.ex, fL.nm0123)) return;
     if (!d_push_nm(cfg.nm_cnt, cfg.ex, fR.nm0123)) return;
     cfg.empty_area &= ~fL.shape;
     cfg.empty_area &= ~fR.shape;
     solutions[atomicAdd(n_solutions, 1)] = cfg; // slice
     if (cfg.height > 1) {
         auto nszid = min(cfg.height - 2, 5);
         auto nsz = d_frowInfoL[cfg.empty_area >> 0 & 0xfu].sz[nszid] *
             d_frowInfoR[cfg.empty_area >> 0 & 0xfu].sz[nszid];
         atomicAdd(n_next, nsz);
     }
}

CudaSearcher::CudaSearcher(uint64_t empty_area)
    : height{ 8 }, n_solutions{ 1 }, solutions{
        new R[]{ empty_area, { ~0u, ~0u, ~0u, ~0u } } {
    if (status() != HOST)
        throw std::runtime_error{ "new solutions weird" };
}

CudaSearcher::~CudaSearcher() {
    if (status() == HOST)
        delete [] solutions;
    else
        C(cudaFree(solutions));
}

void CudaSearcher::search_CPU1() {
    ensure_CPU();
    height--;
    auto solutions = new R[n_next];
    uint64_t n_solutions{}, n_next{};
    for (auto i = 0zu; i < this->n_solutions; i++)
        h_row_search(i, solutions, &n_solutions, &n_next,
                C{ this->solutions[i], height });
}

    C(cudaMallocManaged(&solutions, MAX_SOLUTIONS * sizeof(R)));
    C(cudaMallocManaged(&n_solutions, 2 * sizeof(n_solutions)));
    C(cudaMallocManaged(&n_pending, sizeof(n_pending)));
    C(cudaFree(const_cast<uint32_t *>(n_solutions)));
    C(cudaFree(const_cast<uint32_t *>(n_pending)));
mem_t CudaSearcher::status() const {
    if (!solutions)
        return EMPTY;
    CUmemorytype ret;
    C(cuPointerGetAttribute(&ret, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, solutions));
    return static_cast<mem_t>(ret);
}

void CudaSearcher::invoke_kernel(const R &args) {
    (void)args;
    /*
    for (auto wait = 1u; ; wait = wait >= 1000000u ? 1000000u : 2 * wait) {
        auto fcf_threads = h_fcfs[std::countr_zero(args.empty_area)];
        auto err = cudaSuccess;
#define INV(D) \
        if (args.d == D) { \
            searcher_impl<D><<<1, fcf_threads>>>( \
                args.empty_area, solutions, \
                const_cast<uint32_t *>(n_solutions), \
                const_cast<uint32_t *>(n_pending), \
                args.ex[0], args.ex[1], args.ex[2], args.ex[3], \
                args.ex[4], args.ex[5], args.ex[6]); \
            err = cudaPeekAtLastError(); }
        // INV( 0) INV( 1) INV( 2) INV( 3)
        // INV( 4) INV( 5) INV( 6) INV( 7)
        // INV( 8) INV( 9) INV(10) INV(11)
        // INV(12) INV(13) INV(14) INV(15)
        // INV(16) INV(17) INV(18) INV(19)
        // INV(20) INV(21) INV(22) INV(23)
        // INV(24) INV(25) INV(26) INV(27)
#undef INV
        if (err == cudaErrorLaunchPendingCountExceeded) {
            std::cerr << '.';
            continue;
        }
        C(err);
        // no need to worry about ordering - we are the host thread
        cuda::atomic_ref pd{ *const_cast<uint32_t *>(n_pending) };
        auto p = pd += fcf_threads;
        std::cerr << std::format("p={:10} D{:2} ea{:02} <<<1, {}>>> k={}\n",
                p, args.d, std::popcount(args.empty_area), fcf_threads, n_kernel_invoked);
        return;
    }
    */
}

const unsigned char *CudaSearcher::next() {
    return nullptr;
    /*
    auto flag = false;
    // auto old_val = 0u;
    // auto o = *n_pending;
    while (true) {
    // for (auto wait = 1u; ; wait = wait >= 1000000u ? 1000000u : 2 * wait) {
        auto curr = n_solutions[1];
    again:
        if (curr > n_solution_processed) {
            auto &ret = solutions[n_solution_processed++ % MAX_SOLUTIONS];
            if (ret.d == 0x5555aaaau) {
                auto err = static_cast<cudaError_t>(ret.empty_area);
                throw std::runtime_error{ std::format("{} at #{}: {}",
                        cudaGetErrorName(err), curr, cudaGetErrorString(err)) };
            } else if (ret.d == 0xaaaa5555u) {
                std::cerr << std::format("ret.empty_area = 0x{:016x}\n", ret.empty_area);
                for (auto i = 0; i < 7; i++)
                    std::cerr << std::format("ret.ex[{0}] = 0x{1:08x} = {1}\n", i, ret.ex[i]);
                std::cerr << std::format("ret.d = 0x{:08x}\n", ret.d);
                throw std::runtime_error{ "other error" };
            } else if (ret.empty_area) {
                invoke_kernel(ret);
                flag = false;
                goto again;
            }
            return reinterpret_cast<unsigned char *>(ret.ex);
        }
        if (flag)
            return nullptr;
        auto val = *n_pending;
        if (!val) {
            flag = true;
            continue;
        }
        // if (wait >= 10000000u && val != old_val) {
        //     std::cerr << std::format("waiting, n_pending = {}\n", val);
        //     old_val = val;
        // }
        // usleep(wait);
        usleep(10000);
        std::cerr << std::format("waiting, n_pending = {}\n", val);
    }
    */
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
