#pragma once

#include <cstddef>

template <typename T>
struct alignas(64) Rg {
    T *ptr;
    unsigned long long len; // number of T
};
static_assert(sizeof(Rg<char>) == 64);

template <typename T>
class Growable;
