#pragma once

#include <type_traits>
#include "record.h"

enum class RgType {
    NONE,
    DELETE,
    CUDA_FREE,
};

template <typename T>
struct Rg {
    T *ptr{};
    unsigned long long len{}; // number of T

    RgType ty{ RgType::DELETE };

    [[nodiscard]] operator bool() const { return ptr; }

    void dispose(); // len is not changed
};

struct WL : Rg<R> {
    unsigned pos;
};

static_assert(std::is_trivially_copyable_v<WL>);

extern template void Rg<R>::dispose();
extern template void Rg<RX>::dispose();
