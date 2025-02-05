#include "frow.h"
#include "util.cuh"

int n_devices;

frow_t *d_frowDataL[128][16], *d_frowDataR[128][16];

void transfer_frow_to_gpu() {
    C(cudaGetDeviceCount(&n_devices));
    n_devices = min(n_devices, 1); // FIXME
    std::cout << std::format("n_devices = {}\n", n_devices);
    if (!n_devices)
        throw std::runtime_error{ "no CUDA device" };

    for (auto d = 0; d < n_devices; d++) {
        C(cudaSetDevice(d));
        for (auto i = 0; i < 16; i++) {
            C(cudaMalloc(&d_frowDataL[d][i], h_frowInfoL[i].sz[5] * sizeof(frow_t)));
            C(cudaMalloc(&d_frowDataR[d][i], h_frowInfoR[i].sz[5] * sizeof(frow_t)));
            C(cudaMemcpyAsync(d_frowDataL[d][i], h_frowInfoL[i].data,
                        h_frowInfoL[i].sz[5] * sizeof(frow_t), cudaMemcpyHostToDevice));
            C(cudaMemcpyAsync(d_frowDataR[d][i], h_frowInfoR[i].data,
                        h_frowInfoR[i].sz[5] * sizeof(frow_t), cudaMemcpyHostToDevice));
        }
        C(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, ~0ull));
        size_t drplc;
        C(cudaDeviceGetLimit(&drplc, cudaLimitDevRuntimePendingLaunchCount));
        std::cout << std::format("dev{}.DRPLC = {}\n", d, drplc);
    }
}

void show_gpu_devices() {
  int devs;
  C(cudaGetDeviceCount(&devs));

  printf("Number of devices: %d\n", devs);

  for (int i = 0; i < devs; i++) {
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
