#pragma once

#include <cstddef>

template <typename T>
struct Rg {
    T *ptr;
    unsigned long long len; // number of T
};
