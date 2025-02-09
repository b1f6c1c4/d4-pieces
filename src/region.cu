#include "region.h"
#include "util.cuh"

#include <atomic>

template <typename T>
Rg<T>::operator bool() const {
    std::atomic_ref atm{ ptr };
    return atm.load(std::memory_order_relaxed);
}

template <typename T>
void Rg<T>::dispose() {
    if (!*this)
        return;

    std::atomic_ref atm{ ptr };

    switch (ty) {
        case RgType::NONE:
            break;
        case RgType::DELETE:
            delete ptr;
            atm.store(nullptr, std::memory_order_relaxed);
            break;
        case RgType::CUDA_FREE:
            C(cudaFree(ptr));
            atm.store(nullptr, std::memory_order_relaxed);
            break;
        case RgType::CUDA_FREE_HOST:
            C(cudaFreeHost(ptr));
            atm.store(nullptr, std::memory_order_relaxed);
            break;
        case RgType::CUDA_HOST_UNREGISTER_DELETE:
            C(cudaHostUnregister(ptr));
            delete [] ptr;
            atm.store(nullptr, std::memory_order_relaxed);
            break;
    }
}

template <typename T>
Rg<T> Rg<T>::make_cpu(size_t len, bool page) {
    if (page)
        return Rg<T>{
            reinterpret_cast<T *>(std::aligned_alloc(4096, len * sizeof(RX))),
            CYC_CHUNK,
            RgType::DELETE,
        };
    return Rg<T>{
        new T[len],
        len,
        RgType::DELETE,
    };
}

template <typename T>
Rg<T> Rg<T>::make_managed(size_t len) {
    Rg<T> r{ nullptr, len, RgType::CUDA_FREE };
    C(cudaMallocManaged(&r.ptr, len * sizeof(T)));
    return r;
}

template <typename T>
Rg<T> Rg<T>::make_cuda_mlocked(size_t len, bool direct) {
    if (direct) {
        Rg<T> r{ nullptr, len, RgType::CUDA_FREE_HOST };
        C(cudaHostAlloc(&r.ptr, len * sizeof(T), cudaHostAllocWriteCombined));
        return r;
    } else {
        Rg<T> r{ new T[len], len, RgType::CUDA_HOST_UNREGISTER_DELETE };
        C(cudaHostRegister(r.ptr, len * sizeof(T), cudaHostRegisterReadOnly));
        return r;
    }
}

template void Rg<R>::dispose();
template void Rg<RX>::dispose();
template Rg<R>::operator bool() const;
template Rg<RX>::operator bool() const;
template Rg<R> Rg<R>::make_cpu(size_t len, bool page);
template Rg<RX> Rg<RX>::make_cpu(size_t len, bool page);
template Rg<R> Rg<R>::make_managed(size_t len);
template Rg<RX> Rg<RX>::make_managed(size_t len);
template Rg<R> Rg<R>::make_cuda_mlocked(size_t len, bool direct);
template Rg<RX> Rg<RX>::make_cuda_mlocked(size_t len, bool direct);
