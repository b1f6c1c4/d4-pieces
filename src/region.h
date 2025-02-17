#pragma once

#include <type_traits>
#include "record.h"

enum class RgType {
    NONE,
    DELETE,
    CUDA_FREE,
    CUDA_FREE_HOST,
    CUDA_HOST_UNREGISTER_DELETE,
};

template <typename T>
struct Rg {
    T *ptr{}; // atomic dispose/operator bool is supported

    unsigned long long len{}; // number of T

    RgType ty{ RgType::DELETE };

    [[nodiscard]] bool device_accessible() const { return ty != RgType::DELETE; }
    [[nodiscard]] operator bool() const;

    void dispose(); // len is not changed

    static Rg<T> make_cpu(size_t len, bool page = false);
    static Rg<T> make_managed(size_t len);
    static Rg<T> make_cuda_mlocked(size_t len, bool direct = false, bool h2d = true);

    auto begin() { return ptr; }
    auto end() { return ptr + len; }
};

struct WL : Rg<R> {
    unsigned pos;
};

static_assert(std::is_trivially_copyable_v<WL>);

extern template void Rg<R>::dispose();
extern template void Rg<RX>::dispose();
extern template Rg<R>::operator bool() const;
extern template Rg<RX>::operator bool() const;
extern template Rg<R> Rg<R>::make_cpu(size_t len, bool page);
extern template Rg<RX> Rg<RX>::make_cpu(size_t len, bool page);
extern template Rg<R> Rg<R>::make_managed(size_t len);
extern template Rg<RX> Rg<RX>::make_managed(size_t len);
extern template Rg<R> Rg<R>::make_cuda_mlocked(size_t len, bool direct, bool h2d);
extern template Rg<RX> Rg<RX>::make_cuda_mlocked(size_t len, bool direct, bool h2d);
