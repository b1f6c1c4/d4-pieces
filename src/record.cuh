#pragma once

#include "record.h"

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
