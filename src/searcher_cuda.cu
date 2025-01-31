#include "searcher_cuda.h"

#include <cuda/atomic>

#include <iostream>
#include <format>
#include <unistd.h>
#include <cstdio>

#define MAX_FCFS 384
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

__managed__ static tt_t fcf[64 * MAX_FCFS];
__device__ static unsigned fcfs[64];
static unsigned h_fcfs[64];

void fcf_cache(unsigned pos, const tt_t *data, size_t len) {
    if (len > MAX_FCFS)
        throw std::runtime_error{ std::format("fcf too large {}/{}", len, MAX_FCFS) };
    C(cudaMemcpyToSymbol(fcf, data, len * sizeof(tt_t), pos * MAX_FCFS * sizeof(tt_t)));
    C(cudaMemcpyToSymbol(fcfs, &len, sizeof(unsigned), pos * sizeof(unsigned)));
    h_fcfs[pos] = len;
    if (!pos) {
        C(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, ~0ull));
        size_t drplc;
        C(cudaDeviceGetLimit(&drplc, cudaLimitDevRuntimePendingLaunchCount));
        std::cout << std::format("DRPLC = {}\n", drplc);
    }
}

template <unsigned D> // 0 ~ 27
__global__
void searcher_impl(uint64_t empty_area,
        CudaSearcher::R *solutions, uint32_t *n_solutions, uint32_t *n_pending,
        uint32_t ex0, uint32_t ex1, uint32_t ex2, uint32_t ex3,
        uint32_t ex4, uint32_t ex5, uint32_t ex6) {
    uint64_t nms{}, nmm{}, shape{}, next{};
    uint32_t nmx{};
    uint8_t nm{};

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto pos = __ffsll(empty_area) - 1;
    if (idx >= fcfs[pos]) goto fin;
    shape = fcf[pos * MAX_FCFS + idx].shape;
    nm = fcf[pos * MAX_FCFS + idx].nm;
    if (!(shape & empty_area & -empty_area)) [[unlikely]] {
        auto sptr = atomicAdd(n_solutions, 1);
        auto &ret = solutions[sptr % MAX_SOLUTIONS];
        ret.empty_area = empty_area;
        ret.ex[0] = shape >> 32;
        ret.ex[1] = shape;
        ret.ex[2] = ((uint64_t)&fcf[pos * MAX_FCFS + idx].shape) >> 32;
        ret.ex[3] = ((uint64_t)&fcf[pos * MAX_FCFS + idx].shape);
        ret.ex[4] = pos;
        ret.ex[5] = idx;
        ret.ex[6] = (fcf[pos * MAX_FCFS + idx].shape & 0xffffffffull);
        ret.d = 0xaaaa5555;
        atomicAdd_system(n_solutions + 1, 1);
        goto fin;
    }
    if (shape & ~empty_area) [[likely]] goto fin;
    nmx = __byte_perm(nm, 0, 0); // nm nm nm nm
    if constexpr (D) {
        if constexpr (D >  0) if (__vcmpeq4(nmx, ex0)) [[unlikely]] goto fin;
        if constexpr (D >  4) if (__vcmpeq4(nmx, ex1)) [[unlikely]] goto fin;
        if constexpr (D >  8) if (__vcmpeq4(nmx, ex2)) [[unlikely]] goto fin;
        if constexpr (D > 12) if (__vcmpeq4(nmx, ex3)) [[unlikely]] goto fin;
        if constexpr (D > 16) if (__vcmpeq4(nmx, ex4)) [[unlikely]] goto fin;
        if constexpr (D > 20) if (__vcmpeq4(nmx, ex5)) [[unlikely]] goto fin;
        if constexpr (D > 24) if (__vcmpeq4(nmx, ex6)) [[unlikely]] goto fin;
    }
    nms = static_cast<uint64_t>(nm) << (D % 4) * 8;
    nmm = static_cast<uint64_t>(0xff) << (D % 4) * 8;
         if constexpr (D <  4) ex0 = ((ex0 & ~nmm) | nms);
    else if constexpr (D <  8) ex1 = ((ex1 & ~nmm) | nms);
    else if constexpr (D < 12) ex2 = ((ex2 & ~nmm) | nms);
    else if constexpr (D < 16) ex3 = ((ex3 & ~nmm) | nms);
    else if constexpr (D < 20) ex4 = ((ex4 & ~nmm) | nms);
    else if constexpr (D < 24) ex5 = ((ex5 & ~nmm) | nms);
    else if constexpr (D < 28) ex6 = ((ex6 & ~nmm) | nms);
    next = empty_area & ~shape;
    if (!next) {
        auto sptr = atomicAdd(n_solutions, 1);
        auto &ret = solutions[sptr % MAX_SOLUTIONS];
        ret.empty_area = 0u;
        ret.ex[0] = (D >=  0) ? ex0 : ~0u;
        ret.ex[1] = (D >=  4) ? ex1 : ~0u;
        ret.ex[2] = (D >=  8) ? ex2 : ~0u;
        ret.ex[3] = (D >= 12) ? ex3 : ~0u;
        ret.ex[4] = (D >= 16) ? ex4 : ~0u;
        ret.ex[5] = (D >= 20) ? ex5 : ~0u;
        ret.ex[6] = (D >= 24) ? ex6 : ~0u;
        ret.d = D;
        atomicAdd_system(n_solutions + 1, 1);
        goto fin;
    }
    if constexpr (D < 28) {
        auto err = cudaErrorLaunchPendingCountExceeded;
        auto fcf_threads = fcfs[__ffsll(next)];
        auto dp = true;
        if (dp) {
            atomicAdd_system(n_pending, fcf_threads);
            searcher_impl<D + 1><<<1, fcf_threads, 0, cudaStreamFireAndForget>>>(
                    next, solutions, n_solutions, n_pending,
                    ex0, ex1, ex2, ex3, ex4, ex5, ex6);
            err = cudaPeekAtLastError();
            if (err == cudaSuccess)
                goto fin;
        }
        auto sptr = atomicAdd(n_solutions, 1);
        auto &ret = solutions[sptr % MAX_SOLUTIONS];
        ret.ex[0] = (D >=  0) ? ex0 : ~0u;
        ret.ex[1] = (D >=  4) ? ex1 : ~0u;
        ret.ex[2] = (D >=  8) ? ex2 : ~0u;
        ret.ex[3] = (D >= 12) ? ex3 : ~0u;
        ret.ex[4] = (D >= 16) ? ex4 : ~0u;
        ret.ex[5] = (D >= 20) ? ex5 : ~0u;
        ret.ex[6] = (D >= 24) ? ex6 : ~0u;
        if (err == cudaErrorLaunchPendingCountExceeded) {
            ret.empty_area = empty_area & ~shape;
            ret.d = D + 1;
        } else {
            ret.empty_area = err;
            ret.d = 0x5555aaaa;
        }
        atomicAdd_system(n_solutions + 1, 1);
        if (dp)
            atomicSub_system(n_pending, fcf_threads);
        goto fin;
    }
fin:;
    atomicSub_system(n_pending, 1);
}

CudaSearcher::CudaSearcher(size_t num_shapes)
    : solutions{}, n_solutions{}, n_solution_processed{}, n_kernel_invoked{}, n_pending{} {
    C(cudaMallocManaged(&solutions, MAX_SOLUTIONS * sizeof(R)));
    C(cudaMallocManaged(&n_solutions, 2 * sizeof(n_solutions)));
    C(cudaMallocManaged(&n_pending, sizeof(n_pending)));
}

CudaSearcher::~CudaSearcher() {
    C(cudaFree(solutions));
    C(cudaFree(const_cast<uint32_t *>(n_solutions)));
    C(cudaFree(const_cast<uint32_t *>(n_pending)));
}

void CudaSearcher::start_search(uint64_t empty_area) {
    n_solutions[0] = 0;
    n_solutions[1] = 0;
    *n_pending = 0;
    invoke_kernel(R{ empty_area, 0u, { ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u } });
}

void CudaSearcher::invoke_kernel(const R &args) {
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
        INV( 0) INV( 1) INV( 2) INV( 3)
        INV( 4) INV( 5) INV( 6) INV( 7)
        INV( 8) INV( 9) INV(10) INV(11)
        INV(12) INV(13) INV(14) INV(15)
        INV(16) INV(17) INV(18) INV(19)
        INV(20) INV(21) INV(22) INV(23)
        INV(24) INV(25) INV(26) INV(27)
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
}

const unsigned char *CudaSearcher::next() {
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
  }
}
