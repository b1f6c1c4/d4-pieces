#pragma once

#include <cstdint>

template <unsigned S = 0> // 0,1,2,3
__device__ __forceinline__
void d_wave0(uint32_t cnt, uint32_t v[4]) {
#pragma unroll
    for (auto o = 0; o < 4; o += 2) {
        if (4 * o >= cnt) return;
        unsigned pa, pb;
        switch (S) {
            case 0: pa = 0x7430u, pb = 0x6521u; break;
            case 1: pa = 0x7520u, pb = 0x6431u; break;
            case 2:
                if (o == 2)
                    pa = 0x7531u, pb = 0x6420u;
                else
                    pa = 0x6420u, pb = 0x7531u;
                break;
            case 3: pa = 0x6420u, pb = 0x7531u; break;
            default: __builtin_unreachable();
        }
        auto cmp = __vcmpgtu4(
                __byte_perm(v[o + 0], v[o + 1], pa),
                __byte_perm(v[o + 0], v[o + 1], pb));
        v[o + 0] = __byte_perm(v[o + 0], 0, 0x3210u ^ ((cmp >>  0) & 0x1111u));
        v[o + 1] = __byte_perm(v[o + 1], 0, 0x3210u ^ ((cmp >> 16) & 0x1111u));
    }
}

template <unsigned S = 1> // 1,2,3
__device__ __forceinline__
void d_wave1(uint32_t cnt, uint32_t v[4]) {
#pragma unroll
    for (auto o = 0; o < 4; o += 2) {
        if (4 * o >= cnt) break;
        unsigned pa, pb;
        switch (S) {
            case 1: pa = 0x7610u, pb = 0x5432u; break;
            case 2:
                if (o == 2)
                    pa = 0x7632u, pb = 0x5410u;
                else
                    pa = 0x5410u, pb = 0x7632u;
                break;
            case 3: pa = 0x5410u, pb = 0x7632u; break;
            default: __builtin_unreachable();
        }
        auto cmp = __vcmpgtu4(
                __byte_perm(v[o + 0], v[o + 1], pa),
                __byte_perm(v[o + 0], v[o + 1], pb));
        v[o + 0] = __byte_perm(__byte_perm(v[o + 0], 0,
                    0x3120u ^ ((cmp >>  0) & 0x2222u)), 0, 0x3120u);
        v[o + 1] = __byte_perm(__byte_perm(v[o + 1], 0,
                    0x3120u ^ ((cmp >> 16) & 0x2222u)), 0, 0x3120u);
    }
    d_wave0<S>(cnt, v);
}

template <unsigned S = 2> // 2,3
__device__ __forceinline__
void d_wave2(uint32_t cnt, uint32_t v[4]) {
    uint32_t cmp, p, q;
    cmp = __vcmpgtu4(v[0], v[1]);
    p = __byte_perm(v[0], v[1], 0x5140u ^ ((cmp >>  0) & 0x4444u));
    q = __byte_perm(v[0], v[1], 0x7362u ^ ((cmp >> 16) & 0x4444u));
    v[0] = __byte_perm(p, q, 0x6420u), v[1] = __byte_perm(p, q, 0x7531u);
    cmp = S == 2 ? __vcmpgtu4(v[3], v[2]) : __vcmpgtu4(v[2], v[3]);
    p = __byte_perm(v[2], v[3], 0x5140u ^ ((cmp >>  0) & 0x4444u));
    q = __byte_perm(v[2], v[3], 0x7362u ^ ((cmp >> 16) & 0x4444u));
    v[2] = __byte_perm(p, q, 0x6420u), v[3] = __byte_perm(p, q, 0x7531u);
    d_wave1<S>(cnt, v);
}

__device__ __forceinline__
void d_wave3(uint32_t cnt, uint32_t v[4]) {
    uint32_t cmp, p, q;
    cmp = __vcmpgtu4(v[0], v[2]);
    p = __byte_perm(v[0], v[2], 0x5140u ^ ((cmp >>  0) & 0x4444u));
    q = __byte_perm(v[0], v[2], 0x7362u ^ ((cmp >> 16) & 0x4444u));
    v[0] = __byte_perm(p, q, 0x6420u), v[2] = __byte_perm(p, q, 0x7531u);
    cmp = __vcmpgtu4(v[1], v[3]);
    p = __byte_perm(v[1], v[3], 0x5140u ^ ((cmp >>  0) & 0x4444u));
    q = __byte_perm(v[1], v[3], 0x7362u ^ ((cmp >> 16) & 0x4444u));
    v[1] = __byte_perm(p, q, 0x6420u), v[3] = __byte_perm(p, q, 0x7531u);
    d_wave2<3>(cnt, v);
}

__device__ __forceinline__
void d_sn(uint32_t cnt, uint32_t v[4]) {
    if (cnt <= 0u) return;
    d_wave0(cnt, v);
    d_wave1(cnt, v);
    if (cnt <= 4u) return;
    d_wave2(cnt, v);
    if (cnt <= 8u) return;
    d_wave3(cnt, v);
}

__device__ __forceinline__
bool d_uniq_chk(uint32_t cnt, const uint32_t v[4]) {
    auto eq = 0;
#pragma unroll
    for (auto o = 0; o < 4; o++) {
        if (4 * o >= cnt) {
            eq += 32 * (4 - o);
            break;
        }
        eq += __popc(__vcmpeq4(v[o], __byte_perm(
                    v[o], o < 3 ? v[o + 1] : 0xffffffffu, 0x4321u)));
    }
    return eq == (16 - cnt) * 8;
}

__device__ __forceinline__
void d_push(uint32_t &cnt, uint32_t v[4], uint32_t nv) {
    auto mask = __vcmpne4(nv, 0xffffffffu);
    auto n = __popc(mask);
    if (cnt % 4 == 0) {
        v[cnt / 4] = nv;
    } else {
        auto &p = v[cnt / 4];
        switch (cnt % 4) {
            case 1: p = __byte_perm(p, nv, 0x6540u); break;
            case 2: p = __byte_perm(p, nv, 0x5410u); break;
            case 3: p = __byte_perm(p, nv, 0x4210u); break;
            default: __builtin_unreachable(); break;
        }
        if (cnt / 4 + 1 < 4) {
            v[cnt / 4 + 1] = ~((~nv) >> (4 - cnt % 4) * 8);
        }
    }
    cnt += n / 8;
}
