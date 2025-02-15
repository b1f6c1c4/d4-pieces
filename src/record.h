#pragma once

#include <cstdint>

struct R {
    uint32_t xaL; // requires parse_R and assemble_R
    uint32_t xaH; // requires parse_R and assemble_R
    uint32_t ex0, ex1, ex2;

    bool operator==(const R &other) const = default;
    [[nodiscard]] constexpr uint8_t get_cnt(unsigned height) const {
        return (height >= 5 ? xaH : xaL) >> 24;
    }
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
    union {
        uint32_t ex[4];
        uint8_t nm[16];
    };
};
