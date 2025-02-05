#pragma once

#include <cstdint>

struct R {
    uint32_t xaL; // requires parse_R and assemble_R
    uint32_t xaH; // requires parse_R and assemble_R
    uint32_t ex0, ex1, ex2;

    bool operator==(const R &other) const = default;
};

struct RX : R {
    uint8_t ea;
};

static_assert(sizeof(R) == 20);
static_assert(alignof(R) == 4);
static_assert(sizeof(RX) == 24);
static_assert(alignof(RX) == 4);

struct RCfg {
    uint64_t empty_area;
    uint32_t nm_cnt;
    uint32_t ex[4];
};

template <typename T>
struct Rg {
    T *ptr;
    unsigned long long len; // number of T
};
