#include "searcher_cuda.h"
#include "growable.cuh"
#include "sn.cuh"

#include <cuda.h>
#include <cuda/atomic>
#include <cstring>
#include <memory>
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

static int n_devices;
static const frow_info_t *h_frowInfoL, *h_frowInfoR;
static frow_t *d_frowDataL[128][16], *d_frowDataR[128][16];

void frow_cache(const frow_info_t *fiL, const frow_info_t *fiR) {
    C(cudaGetDeviceCount(&n_devices));
    n_devices = min(n_devices, 128);
    std::cout << std::format("n_devices = {}\n", n_devices);
    if (!n_devices)
        throw std::runtime_error{ "no CUDA device" };

    h_frowInfoL = fiL;
    h_frowInfoR = fiR;
    for (auto d = 0; d < n_devices; d++) {
        C(cudaSetDevice(d));
        for (auto i = 0; i < 16; i++) {
            C(cudaMalloc(&d_frowDataL[d][i], fiL[i].sz[5] * sizeof(frow_t)));
            C(cudaMalloc(&d_frowDataR[d][i], fiR[i].sz[5] * sizeof(frow_t)));
            C(cudaMemcpyAsync(d_frowDataL[d][i], fiL[i].data,
                        fiL[i].sz[5] * sizeof(frow_t), cudaMemcpyHostToDevice));
            C(cudaMemcpyAsync(d_frowDataR[d][i], fiR[i].data,
                        fiR[i].sz[5] * sizeof(frow_t), cudaMemcpyHostToDevice));
        }
        C(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, ~0ull));
        size_t drplc;
        C(cudaDeviceGetLimit(&drplc, cudaLimitDevRuntimePendingLaunchCount));
        std::cout << std::format("dev{}.DRPLC = {}\n", d, drplc);
    }
}

#define CYC_CHUNK (10ull * 1048576ull / sizeof(R))

__global__
void d_row_search(
        R *bins,
        unsigned long long *n_bins,
        const uint32_t *n_available_chunks,
        uint32_t *n_completed_chunks,
        const R *cfgs, uint64_t n_cfgs,
        const frow_t *f0L, uint32_t f0Lsz,
        const frow_t *f0R, uint32_t f0Rsz) {
    auto idx = threadIdx.x + static_cast<uint64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_cfgs * f0Lsz * f0Rsz) [[unlikely]] return;
    auto cfg = cfgs[idx / f0Rsz / f0Lsz];
    auto fL  = f0L [idx / f0Rsz % f0Lsz];
    auto fR  = f0R [idx % f0Rsz];
    if (fL.shape & ~cfg.empty_area) [[unlikely]] return;
    if (fR.shape & ~cfg.empty_area) [[unlikely]] return;
    if (fL.shape & fR.shape) [[unlikely]] return;
    d_push(cfg.nm_cnt, cfg.ex, fL.nm0123);
    d_push(cfg.nm_cnt, cfg.ex, fR.nm0123);
    d_sn(cfg.nm_cnt, cfg.ex);
    if (!d_uniq_chk(cfg.nm_cnt, cfg.ex)) [[unlikely]] return;
    cfg.empty_area &= ~fL.shape;
    cfg.empty_area &= ~fR.shape;
    __builtin_assume(!(cfg.empty_area & 0b11111111u));
    cfg.empty_area >>= 8;
    auto out = __nv_atomic_fetch_add(n_bins, 1,
            __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE);
    unsigned long long cap;
spin:
    cap = __nv_atomic_load_n(const_cast<uint32_t *>(n_available_chunks),
            __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM) * CYC_CHUNK;
    if (out >= cap)
        goto spin;
    bins[out] = cfg; // slice
    if (out % CYC_CHUNK == 0) {
        __nv_atomic_fetch_add(n_completed_chunks, 1,
                __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM);
    }
}

CudaSearcher::CudaSearcher(uint64_t empty_area)
    : solutions{}, grs{},
      height{ (std::bit_width(empty_area) + 8u - 1u) / 8u } {
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

#define VMEM_SZ ((1zu << 43) / sizeof(R))

struct Device {
    int dev;
    cudaStream_t stream;

    std::unique_ptr<Growable<R>> gr;

    uint32_t *counters;
    uint32_t n_collected_chunks;

    R *bins; // __device__, but owned by Growable<R>
    unsigned long long *n_bins; // __device__, owned

    Device(Device &&other) = default;
    Device &operator=(Device &&other) = default;
    explicit Device(int d)
        : dev{ d }, stream{}, gr{},
          counters{}, n_collected_chunks{}, bins{}, n_bins{} {
        C(cudaMallocManaged(&counters, 2 * sizeof(uint32_t)));

        C(cudaSetDevice(d));
        C(cudaStreamCreate(&stream));
        C(cudaMallocAsync(&n_bins, sizeof(unsigned long long), stream));

        std::cout << std::format("dev#{}: map {}B of vmem, then fill it with {}B chunks\n",
                dev, display(VMEM_SZ), display(CYC_CHUNK * sizeof(R)));
        gr = std::make_unique<Growable<R>>(VMEM_SZ);
        auto k = 0;
        while (true) {
            if (auto p = gr->get(CYC_CHUNK * ++k); p)
                bins = p;
            else
                break;
        }
        k--;
        cuda::atomic_ref n_available_chunks{ counters[0] };
        n_available_chunks.fetch_add(k, cuda::memory_order_release);
        std::cout << std::format("dev#{}: {} * {}B = {}B of mem ({}) mapped\n",
                dev, k, display(CYC_CHUNK * sizeof(R)), display(k * CYC_CHUNK * sizeof(R)), k * CYC_CHUNK);
    }

    ~Device() {
        C(cudaSetDevice(dev));
        C(cudaFreeAsync(counters, stream));
        C(cudaFreeAsync(n_bins, stream));
        C(cudaStreamSynchronize(stream));
        C(cudaStreamDestroy(stream));
    }

    [[nodiscard]] bool ready() const {
        auto res = cudaStreamQuery(stream);
        switch (res) {
            case cudaSuccess: return true;
            case cudaErrorNotReady: return false;
            default: C(res); return false;
        }
    }

    void join() {
        std::cout << std::format("dev#{}: synchronize\n", dev);
        C(cudaStreamSynchronize(stream));
    }

    void dispatch(unsigned pos, unsigned height, Rg<R> cfgs) {
        auto [ptr, len] = cfgs;
        auto szid = min(height - 1, 5);
        auto fanoutL = h_frowInfoL[(pos >> 0) & 0b1111u].sz[szid];
        auto fanoutR = h_frowInfoR[(pos >> 4) & 0b1111u].sz[szid];
        auto sz = len * fanoutL * fanoutR;
        auto d_f0L = d_frowDataL[dev][pos >> 0 & 0xfu];
        auto d_f0R = d_frowDataR[dev][pos >> 4 & 0xfu];
        auto [b, t] = balance(sz);
        std::cout << std::format("#{:08b}-dev#{}: <<<{}, {}>>> = {} * L{} * R{} => max {}B\n",
                pos, dev, b, t, len, fanoutL, fanoutR, display(sz * sizeof(R)));
        C(cudaSetDevice(dev));
        C(cudaStreamAttachMemAsync(stream, ptr, len * sizeof(R)));
        C(cudaMemPrefetchAsync(ptr, len * sizeof(R), dev, stream));
        d_row_search<<<b, t, 0, stream>>>(bins, n_bins,
                &counters[0], &counters[1],
                ptr, len,
                d_f0L, fanoutL,
                d_f0R, fanoutR);
        C(cudaFreeAsync(ptr, stream));
    }

    void collect(Sorter &sorter, bool force = false) {
        cuda::atomic_ref n_available_chunks{ counters[0] };
        cuda::atomic_ref n_completed_chunks{ counters[1] };
        auto completed = n_completed_chunks.load(cuda::memory_order_acquire);
        auto k = completed > n_collected_chunks ? completed - n_collected_chunks : 0u;
        if (k) {
            auto n = n_available_chunks.load(cuda::memory_order_relaxed);
            std::cout << std::format("dev#{}: {}/{} (+{}) chunks ({}B) recycled\n",
                    dev, completed, n, k, display(k * CYC_CHUNK * sizeof(R)));
        }
        for (auto i = 0u; i < k; i++) {
            gr->commit(CYC_CHUNK);
            gr->evict1();
            n_available_chunks.fetch_add(1, cuda::memory_order_acquire);
            completed++;
        }
        if (!force && !sorter.ready())
            return;
        std::cout << std::format("dev#{}: pushing {} entries to sorter ({}B)\n",
                dev, gr->get_load(), display(gr->get_load() * sizeof(R)));
        sorter.push(gr->remove_data());
    }
};

void CudaSearcher::search_GPU() {
    Sorter sorter{ *this };
    std::vector<Device> devs;
    for (auto i = 0; i < n_devices; i++)
        devs.emplace_back(i);

    for (unsigned ipos = 0u, dev = 0u; ipos <= 255u; ipos++) {
        devs[dev].dispatch(ipos, height, solutions[ipos]);
        dev = (dev + 1) % n_devices;
    }
    bool flag;
    do {
        flag = true;
        for (auto &dev : devs) {
            flag &= dev.ready();
            dev.collect(sorter);
        }
    } while (!flag);
    for (auto &dev : devs)
        dev.collect(sorter, true);
    devs.clear();
    sorter.join();
}

uint64_t CudaSearcher::next_size(unsigned pos) const {
    auto szid = min(height - 1, 5);
    return solutions[pos].len
        * h_frowInfoL[(pos >> 0) & 0b1111u].sz[szid]
        * h_frowInfoR[(pos >> 4) & 0b1111u].sz[szid];
}

Rg<R> CudaSearcher::write_solution(unsigned pos, size_t sz) {
    auto &r = solutions[pos];
    C(cudaFree(r.ptr));
    r.ptr = nullptr, r.len = 0;
    C(cudaMallocManaged(&r.ptr, sz * sizeof(R), cudaMemAttachHost));
    r.len = sz;
    return r;
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
