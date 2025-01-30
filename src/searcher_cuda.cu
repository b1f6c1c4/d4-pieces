#include "searcher_cuda.h"

#include <iostream>
#include <format>
#include <unistd.h>
#include <cstdio>

#define MAX_FCFS 16384
#define MAX_SOLUTIONS 1048576
#define fcf_threads 64

#define C(ans) { chk_impl((ans), __FILE__, __LINE__); }

static inline void chk_impl(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << std::format("CUDA: {} @ {}:{}\n", cudaGetErrorString(code), file, line);
    }
}

__device__ static tt_t fcf[MAX_FCFS];
__device__ static size_t shps;
__device__ static size_t fcfs;

void fcf_cache(size_t num_shapes) {
    C(cudaMemcpyToSymbol(fcf, fast_canonical_form, fast_canonical_forms * sizeof(tt_t)));
    C(cudaMemcpyToSymbol(shps, &num_shapes, sizeof(size_t)));
    C(cudaMemcpyToSymbol(fcfs, &fast_canonical_forms, sizeof(size_t)));
    C(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, ~0ull));
    size_t drplc;
    C(cudaDeviceGetLimit(&drplc, cudaLimitDevRuntimePendingLaunchCount));
    std::cout << std::format("DRPLC = {}\n", drplc);
}

template <unsigned D> // 0 ~ 31
__global__
void searcher_impl(uint64_t empty_area,
        uint32_t (*solutions)[8], uint32_t *n_solutions, uint32_t *n_pending,
        uint32_t ex0, uint32_t ex1, uint32_t ex2, uint32_t ex3,
        uint32_t ex4, uint32_t ex5, uint32_t ex6, uint32_t ex7) {
    uint64_t nms{}, nmm{}, covering{}, shape{};
    uint32_t nmx{};
    uint8_t nm{};

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= fcfs) goto fin;
    covering = empty_area & -empty_area; // Shape<8>::front_shape();
    shape = fcf[idx].shape;
    nm = fcf[idx].nm;
    if (!(shape & covering)) [[likely]] goto fin;
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
        if constexpr (D > 28) if (__vcmpeq4(nmx, ex7)) [[unlikely]] goto fin;
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
    else if constexpr (D < 32) ex7 = ((ex7 & ~nmm) | nms);
    if (!(empty_area & ~shape)) {
        auto pos = atomicAdd(n_solutions, 1);
        auto ptr = solutions[pos];
        if (pos >= MAX_SOLUTIONS)
            goto fin;
        ptr[0] = (D >=  0) ? ex0 : ~0u;
        ptr[1] = (D >=  4) ? ex1 : ~0u;
        ptr[2] = (D >=  8) ? ex2 : ~0u;
        ptr[3] = (D >= 12) ? ex3 : ~0u;
        ptr[4] = (D >= 16) ? ex4 : ~0u;
        ptr[5] = (D >= 20) ? ex5 : ~0u;
        ptr[6] = (D >= 24) ? ex6 : ~0u;
        ptr[7] = (D >= 28) ? ex7 : ~0u;
        goto fin;
    }
    if constexpr (D < 32) {
        auto fcf_blocks = (fcfs + fcf_threads - 1) / fcf_threads;
    again:
        searcher_impl<D + 1><<<fcf_blocks, fcf_threads, 0, cudaStreamFireAndForget>>>(
                empty_area & ~shape, solutions, n_solutions, n_pending,
                ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
        auto err = cudaPeekAtLastError();
        if (err == cudaErrorLaunchPendingCountExceeded) {
            __nanosleep(~0u);
            goto again;
        }
        if (err != cudaSuccess) {
            auto pos = atomicAdd(n_solutions, 1);
            auto ptr = solutions[pos];
            if (pos >= MAX_SOLUTIONS)
                goto fin;
            ptr[0] = 0x5555aaaau;
            ptr[1] = err;
            ptr[2] = 0xaaaa5555u;
            goto fin;
        }
        atomicAdd(n_pending, fcf_blocks * fcf_threads);
    }
fin:
    atomicSub(n_pending, 1);
}

CudaSearcher::CudaSearcher(size_t num_shapes)
    : solutions{}, n_solutions{}, n_solution_processed{}, n_pending{} {
    C(cudaMallocManaged(&solutions, MAX_SOLUTIONS * sizeof(*solutions)));
    C(cudaMallocManaged(&n_solutions, sizeof(n_solutions)));
    C(cudaMallocManaged(&n_pending, sizeof(n_pending)));
}

CudaSearcher::~CudaSearcher() {
    C(cudaFree(solutions));
    C(cudaFree(const_cast<uint32_t *>(n_solutions)));
    C(cudaFree(const_cast<uint32_t *>(n_pending)));
}

void CudaSearcher::start_search(uint64_t empty_area) {
    auto fcf_blocks = (fast_canonical_forms + fcf_threads - 1) / fcf_threads;
    *n_solutions = 0;
    *n_pending = fcf_blocks * fcf_threads;
    searcher_impl<0><<<fcf_blocks, fcf_threads>>>(empty_area,
        reinterpret_cast<uint32_t (*)[8]>(solutions),
        const_cast<uint32_t *>(n_solutions),
        const_cast<uint32_t *>(n_pending),
        ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u);
    C(cudaPeekAtLastError());
}

const unsigned char *CudaSearcher::next() {
    auto flag = false;
    // auto o = *n_pending;
    for (auto wait = 1u; ; wait = wait >= 1000000u ? 1000000u : 2 * wait) {
        auto curr = *n_solutions;
        if (curr > n_solution_processed)
            return solutions[n_solution_processed++];
        if (flag)
            return nullptr;
        if (!*n_pending) {
            flag = true;
            continue;
        }
        usleep(wait);
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
  }
}
