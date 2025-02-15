#include "frow.h"
#include "util.cuh"

int n_devices;
frow_info_d d_frowDataL[128][16], d_frowDataR[128][16];

void transfer_frow_to_gpu() {
    C(cudaGetDeviceCount(&n_devices));
    n_devices = min(n_devices, 128);
    if (!n_devices)
        THROW("no CUDA device");

    for (auto d = 0; d < n_devices; d++) {
        C(cudaSetDevice(d));
        for (auto i = 0; i < 16; i++) {
#define CP(X, ty, field) \
            C(cudaMalloc(&d_frowData ## X[d][i].field, h_frowInfo ## X[i].sz[5] * sizeof(ty))); \
            C(cudaMemcpyAsync(d_frowData ## X[d][i].field, h_frowInfo ## X[i].field, \
                        h_frowInfo ## X[i].sz[5] * sizeof(ty), cudaMemcpyHostToDevice));
            CP(L, frow32_t, data32)
            CP(L, uint32_t, dataL)
            CP(L, uint32_t, dataH)
            CP(L, uint32_t, data0123)
            CP(R, frow32_t, data32)
            CP(R, uint32_t, dataL)
            CP(R, uint32_t, dataH)
            CP(R, uint32_t, data0123)
#undef CP
        }
    }
    for (auto d = 0; d < n_devices; d++) {
        C(cudaSetDevice(d));
        C(cudaDeviceSynchronize());
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
    printf("  PMA w/ host PT: %s\n",prop.pageableMemoryAccessUsesHostPageTables ? "yes" : "no");
    printf("  ECC: %s\n",prop.ECCEnabled ? "yes" : "no");
    printf("  Cooperative launch: %s\n",prop.cooperativeLaunch ? "yes" : "no");
    printf("  DMMA from host: %s\n",prop.directManagedMemAccessFromHost ? "yes" : "no");
    printf("  L2 Cache Size (B): %d\n",prop.l2CacheSize);
    printf("  Persistting L2 Cache Size (B): %d\n",prop.persistingL2CacheMaxSize);
    printf("  L2 Window (B): %d\n",prop.accessPolicyMaxWindowSize);
    printf("  Shared mem per block (B): %lu\n",prop.sharedMemPerBlock);
    printf("  Shared mem per mp (B): %lu\n",prop.sharedMemPerMultiprocessor);
    printf("  Const mem (B): %lu\n",prop.totalConstMem);
    printf("  Global mem (MiB): %lf\n",prop.totalGlobalMem / 1024.0 / 1024);
    int v;
    cudaDeviceGetAttribute(&v, cudaDevAttrMemSyncDomainCount, i);
    printf("  Sync domain: %d\n",v);
    cudaDeviceGetAttribute(&v, cudaDevAttrHostRegisterSupported, i);
    printf("  Host register: %d\n",v);
    cudaDeviceGetAttribute(&v, cudaDevAttrSingleToDoublePrecisionPerfRatio, i);
    printf("  float/double ratio: %d\n", v);
    cudaDeviceGetAttribute(&v, (cudaDeviceAttr)102, i);
    printf("  VMM: %d\n", v);
  }
}
