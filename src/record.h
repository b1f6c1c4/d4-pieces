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

#ifdef __CUDA_ARCH__
template <unsigned H> __device__ RX assemble_R(RCfg rc);
template <unsigned H> __device__ RCfg  parse_R(R cfg, uint8_t ea);
extern template __device__ RX assemble_R<0>(RCfg rc);
extern template __device__ RCfg  parse_R<1>(R cfg, uint8_t ea);
extern template __device__ RX assemble_R<1>(RCfg rc);
extern template __device__ RCfg  parse_R<2>(R cfg, uint8_t ea);
extern template __device__ RX assemble_R<2>(RCfg rc);
extern template __device__ RCfg  parse_R<3>(R cfg, uint8_t ea);
extern template __device__ RX assemble_R<3>(RCfg rc);
extern template __device__ RCfg  parse_R<4>(R cfg, uint8_t ea);
extern template __device__ RX assemble_R<4>(RCfg rc);
extern template __device__ RCfg  parse_R<5>(R cfg, uint8_t ea);
extern template __device__ RX assemble_R<5>(RCfg rc);
extern template __device__ RCfg  parse_R<6>(R cfg, uint8_t ea);
extern template __device__ RX assemble_R<6>(RCfg rc);
extern template __device__ RCfg  parse_R<7>(R cfg, uint8_t ea);
extern template __device__ RX assemble_R<7>(RCfg rc);
extern template __device__ RCfg  parse_R<8>(R cfg, uint8_t ea);
#endif
