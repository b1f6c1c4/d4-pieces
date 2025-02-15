#pragma once

#include "record.h"

template <unsigned H>
__device__
RCfg parse_R(R cfg, uint8_t ea) {
    static_assert(H <= 8);
    static_assert(H >= 1);
    RCfg rc{ 0ull, 0u, { ~0u, ~0u, ~0u, ~0u } };
    if constexpr (H == 8) {
        rc.empty_area = (cfg.xaL | (uint64_t)cfg.xaH << 32) << 8 | ea;
        return rc;
    }
    if constexpr (H == 7) {
        rc.empty_area = cfg.xaL | (uint64_t)(cfg.xaH & 0x0000ffffu) << 32;
        rc.nm_cnt = cfg.xaH >> 24;
    }
    if constexpr (H == 6) {
        rc.empty_area = cfg.xaL | (uint64_t)(cfg.xaH >> 16 & 0xffu) << 32;
        rc.nm_cnt = cfg.xaH >> 24;
        rc.ex[3] = cfg.xaH | 0xffff0000u;
    }
    if constexpr (H == 5) {
        rc.empty_area = cfg.xaL;
        rc.nm_cnt = cfg.xaH >> 24;
        rc.ex[3] = cfg.xaH | 0xff000000u;
    }
    if constexpr (H <= 4) {
        rc.empty_area = cfg.xaL & 0x00ffffffu;
        rc.nm_cnt = cfg.xaL >> 24;
        rc.ex[3] = cfg.xaH;
    }
    if constexpr (H <= 6)
        rc.ex[2] = cfg.ex2;
    rc.ex[1] = cfg.ex1;
    rc.ex[0] = cfg.ex0;
    rc.empty_area <<= 8;
    rc.empty_area |= ea;
    return rc;
}

template <unsigned H> // height - 1
__device__
RX assemble_R(RCfg rc) {
    static_assert(H <= 7);
    __builtin_assume(!(rc.empty_area & 0b11111111u));
    RX cfg;
    cfg.ea = rc.empty_area >> 8 & 0xffu;
    if constexpr (H == 7) {
        cfg.xaL = rc.empty_area >> 16; 
        cfg.xaH = rc.empty_area >> 48 | rc.nm_cnt << 24; 
    }
    if constexpr (H == 6) {
        cfg.xaL = rc.empty_area >> 16; 
        cfg.xaH = __byte_perm(__byte_perm(rc.ex[3], rc.nm_cnt, 0x4f10u),
                rc.empty_area >> 32, 0x3610u);
    }
    if constexpr (H == 5) {
        cfg.xaL = rc.empty_area >> 16; 
        cfg.xaH = __byte_perm(rc.ex[3], rc.nm_cnt, 0x4210u);
    }
    if constexpr (H <= 4) {
        cfg.xaL = __byte_perm(rc.empty_area >> 16, rc.nm_cnt, 0x4210u);
        cfg.xaH = rc.ex[3];
    }
    if constexpr (H <= 6)
        cfg.ex2 = rc.ex[2];
    cfg.ex1 = rc.ex[1];
    cfg.ex0 = rc.ex[0];
    return cfg;
}
