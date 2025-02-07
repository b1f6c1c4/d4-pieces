#include "region.h"
#include "util.cuh"

template <typename T>
void Rg<T>::dispose() {
    if (!*this)
        return;

    switch (ty) {
        case RgType::NONE:
            break;
        case RgType::DELETE:
            delete ptr;
            ptr = nullptr;
            break;
        case RgType::CUDA_FREE:
            C(cudaFree(ptr));
            ptr = nullptr;
            break;
    }
}

template void Rg<R>::dispose();
template void Rg<RX>::dispose();
