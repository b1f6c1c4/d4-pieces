#include <algorithm>
#include <string>
#include <ranges>
#include <random>
#include <format>
#include <iostream>

#define MAX_PIECES 16
#define MAX_PIECESd4 (MAX_PIECES + 4 - 1) / 4

#define C(ans) { chk_impl((ans), __FILE__, __LINE__); }

static inline void chk_impl(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << std::format("CUDA: {}: {} @ {}:{}\n",
                cudaGetErrorName(code), cudaGetErrorString(code), file, line);
    }
}

__device__ __forceinline__
bool push_nm_mem(uint32_t &nm_cnt, uint8_t mem[256], uint32_t new_nm) {
    auto t = 0;
#pragma unroll
    for (auto v = 0; v < 4; v++) {
        auto nm = (new_nm >> 8 * v) & 0xffu;
        if (nm == 0xffu)
            break;
        if (mem[nm]) [[unlikely]]
            return false;
        t++;
    }
#pragma unroll
    for (auto v = 0; v < 4; v++) {
        auto nm = (new_nm >> 8 * v) & 0xffu;
        mem[nm] = 0xffu;
    }
    nm_cnt += t;
    return true;
}

__device__ __forceinline__
bool push_nm_u8cmp(uint32_t &nm_cnt, uint8_t old_nm[MAX_PIECES], uint32_t new_nm) {
    __builtin_assume(nm_cnt < MAX_PIECES);
#pragma unroll
    for (auto v = 0; v < 4; v++) {
        auto nm = (new_nm >> 8 * v) & 0xffu;
        if (nm == 0xffu)
            break;
#pragma unroll
        for (auto o = 0; o < MAX_PIECES; o++) {
            if (o >= nm_cnt)
                break;
            if (nm == old_nm[o]) [[unlikely]]
                return false;
        }
    }
#pragma unroll
    for (auto v = 0; v < 4; v++) {
        auto nm = (new_nm >> 8 * v) & 0xffu;
        if (nm == 0xffu)
            break;
        old_nm[nm_cnt++] = nm;
    }
    return true;
}

__device__ __forceinline__
bool push_nm_d4cmp(uint32_t &nm_cnt, uint32_t old_nm[MAX_PIECESd4], uint32_t new_nm) {
    __builtin_assume(nm_cnt < MAX_PIECES);
#pragma unroll
    for (auto v = 0; v < 4; v++) {
        auto nm = (new_nm >> 8 * v) & 0xffu;
        if (nm == 0xffu)
            break;
        auto nmx = __byte_perm(new_nm, 0, v << 0 | v << 4 | v << 8 | v << 12);
#pragma unroll
        for (auto o = 0; o < MAX_PIECESd4; o++) {
            if (4 * o >= nm_cnt)
                break;
            if (__vcmpeq4(nmx, old_nm[o])) [[unlikely]]
                return false;
        }
    }
#pragma unroll
    for (auto v = 0; v < 4; v++) {
        auto nm = (new_nm >> 8 * v) & 0xffu;
        if (nm == 0xffu)
            break;
        __builtin_assume(nm_cnt < MAX_PIECES);
        old_nm[nm_cnt / 4] &= ~(0xffu << nm_cnt % 4 * 8);
        old_nm[nm_cnt / 4] |= nm << nm_cnt % 4 * 8;
        nm_cnt++;
    }
    return true;
}

template <typename T>
__device__ __forceinline__
void report_thr(T *out, uint32_t *n_out, T val) {
    auto ptr = out + atomicAdd(n_out, 1);
    *ptr = val;
}

template <typename T>
__device__ __forceinline__
void report_warp(T *out, uint32_t *n_out, T val) {
    auto mask = __ballot_sync(0xffffffffu, 1);
    // if (!mask) return;
    auto first = __ffs(mask) - 1;
    uint32_t cnt0;
    if (threadIdx.x % 32 == first) {
        auto n = __popc(mask);
        cnt0 = atomicAdd_system(n_out, n);
    }
    cnt0 = __shfl_sync(0xffffffffu, cnt0, first);
    auto ptr = out + cnt0 + __popc(mask & ((1u << threadIdx.x % 32) - 1u));
    *ptr = val;
}

#define N_THR 512
#define N_BLK (84 * 3)
#define N_NEWS (N_BLK * N_THR)

template <bool T>
__global__ void check_mem(uint32_t *out, uint32_t *n_out, uint32_t *news) {
    uint32_t nm_cnt{};
    uint8_t mem[256]{};
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (!push_nm_mem(nm_cnt, mem, news[0 * N_NEWS + idx])) return;
    if (!push_nm_mem(nm_cnt, mem, news[1 * N_NEWS + idx])) return;
    if (!push_nm_mem(nm_cnt, mem, news[2 * N_NEWS + idx])) return;
    if (!push_nm_mem(nm_cnt, mem, news[3 * N_NEWS + idx])) return;
    if (!push_nm_mem(nm_cnt, mem, news[4 * N_NEWS + idx])) return;
    if (!push_nm_mem(nm_cnt, mem, news[5 * N_NEWS + idx])) return;
    if (!push_nm_mem(nm_cnt, mem, news[6 * N_NEWS + idx])) return;
    if (!push_nm_mem(nm_cnt, mem, news[7 * N_NEWS + idx])) return;
    if constexpr (T)
        report_thr(out, n_out, nm_cnt);
    else
        report_warp(out, n_out, nm_cnt);
}

template <bool T>
__global__ void check_u8cmp(uint32_t *out, uint32_t *n_out, uint32_t *news) {
    uint32_t nm_cnt{};
    uint8_t old_nm[MAX_PIECES]{};
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (!push_nm_u8cmp(nm_cnt, old_nm, news[0 * N_NEWS + idx])) return;
    if (!push_nm_u8cmp(nm_cnt, old_nm, news[1 * N_NEWS + idx])) return;
    if (!push_nm_u8cmp(nm_cnt, old_nm, news[2 * N_NEWS + idx])) return;
    if (!push_nm_u8cmp(nm_cnt, old_nm, news[3 * N_NEWS + idx])) return;
    if (!push_nm_u8cmp(nm_cnt, old_nm, news[4 * N_NEWS + idx])) return;
    if (!push_nm_u8cmp(nm_cnt, old_nm, news[5 * N_NEWS + idx])) return;
    if (!push_nm_u8cmp(nm_cnt, old_nm, news[6 * N_NEWS + idx])) return;
    if (!push_nm_u8cmp(nm_cnt, old_nm, news[7 * N_NEWS + idx])) return;
    if constexpr (T)
        report_thr(out, n_out, nm_cnt);
    else
        report_warp(out, n_out, nm_cnt);
}

template <bool T>
__global__ void check_d4cmp(uint32_t *out, uint32_t *n_out, uint32_t *news) {
    uint32_t nm_cnt{};
    uint32_t old_nm[MAX_PIECESd4]{ 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu };
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (!push_nm_d4cmp(nm_cnt, old_nm, news[0 * N_NEWS + idx])) return;
    if (!push_nm_d4cmp(nm_cnt, old_nm, news[1 * N_NEWS + idx])) return;
    if (!push_nm_d4cmp(nm_cnt, old_nm, news[2 * N_NEWS + idx])) return;
    if (!push_nm_d4cmp(nm_cnt, old_nm, news[3 * N_NEWS + idx])) return;
    if (!push_nm_d4cmp(nm_cnt, old_nm, news[4 * N_NEWS + idx])) return;
    if (!push_nm_d4cmp(nm_cnt, old_nm, news[5 * N_NEWS + idx])) return;
    if (!push_nm_d4cmp(nm_cnt, old_nm, news[6 * N_NEWS + idx])) return;
    if (!push_nm_d4cmp(nm_cnt, old_nm, news[7 * N_NEWS + idx])) return;
    if constexpr (T)
        report_thr(out, n_out, nm_cnt);
    else
        report_warp(out, n_out, nm_cnt);
}

static uint32_t h_news[8 * N_NEWS];
static uint32_t non_spoiled_cnt = 0;
static uint32_t truth[N_NEWS];

int main(int argc, char *argv[]) {
    using namespace std::string_literals;
    std::mt19937_64 rnd{};
    std::uniform_int_distribution dist{ 0, 50 };
    std::poisson_distribution ndist{ 3.0 };
    std::bernoulli_distribution sdist{ 0.1 };
    std::cout << std::format("generating {} questions\n", N_NEWS);
    for (auto i = 0; i < N_NEWS; i++) {
        auto spoiled = false;
        unsigned used_v{};
        char used[255]{};
        for (auto k = 0; k < 8; k++) {
            auto n = ndist(rnd);
            if (n > 4)
                n = 4;
            if (used_v + n > 16)
                n = 16 - used_v;
            auto nms = 0xffffffffu;
            char k_used[255]{};
            for (auto j = 0; j < n; j++) {
                uint8_t nm;
                do {
                    nm = dist(rnd);
                } while (k_used[nm] || used[nm] && !sdist(rnd));
                if (used[nm]) spoiled = true;
                k_used[nm] = 1;
                used[nm] = 1;
                nms <<= 8, nms |= nm;
            }
            h_news[k * N_NEWS + i] = nms;
            used_v += n;
        }
        if (!spoiled)
            truth[non_spoiled_cnt++] = used_v;
    }
    std::sort(truth, truth + non_spoiled_cnt);

    auto verify = [](uint32_t *answer, uint32_t n_answer) {
        if (n_answer != non_spoiled_cnt) {
            std::cout << std::format("answer count wrong: {} != {}\n", n_answer, non_spoiled_cnt);
            return;
        } else {
            std::cout << std::format("answer count correct: {}/{}\n", non_spoiled_cnt, N_NEWS);
        }
        for (auto i = 0; i < n_answer; i++) {
            if (answer[i] != truth[i])
                std::cout << std::format("answer[0x{:05x}] wrong: 0x{:08x} != 0x{:08x}\n", i, answer[i], truth[i]);
        }
    };

    uint32_t *d_news;
    uint32_t *m_out;
    uint32_t *d_n_out;
    uint32_t h_n_out{};

    std::cout << std::format("copying {:.1f} KiB to GPU\n", 8 * N_NEWS * sizeof(uint32_t) / 1024.0);
    C(cudaMalloc(&d_news, 8 * N_NEWS * sizeof(uint32_t)));
    C(cudaMallocManaged(&m_out, N_NEWS * sizeof(uint32_t)));
    C(cudaMalloc(&d_n_out, sizeof(uint32_t)));
    C(cudaMemcpy(d_news, h_news, 8 * N_NEWS * sizeof(uint32_t), cudaMemcpyHostToDevice));
    C(cudaMemcpy(d_n_out, &h_n_out, sizeof(uint32_t), cudaMemcpyHostToDevice));

    std::cout << std::format("launching kernel <<<{}, {}>>>\n", N_BLK, N_THR);
    if ("mem"s == argv[1]) {
        if ("t"s == argv[2])
            check_mem<true><<<N_BLK, N_THR>>>(m_out, d_n_out, d_news);
        else if ("d"s == argv[2])
            check_mem<false><<<N_BLK, N_THR>>>(m_out, d_n_out, d_news);
    } else if ("u8cmp"s == argv[1]) {
        if ("t"s == argv[2])
            check_u8cmp<true><<<N_BLK, N_THR>>>(m_out, d_n_out, d_news);
        else if ("d"s == argv[2])
            check_u8cmp<false><<<N_BLK, N_THR>>>(m_out, d_n_out, d_news);
    } else if ("d4cmp"s == argv[1]) {
        if ("t"s == argv[2])
            check_d4cmp<true><<<N_BLK, N_THR>>>(m_out, d_n_out, d_news);
        else if ("d"s == argv[2])
            check_d4cmp<false><<<N_BLK, N_THR>>>(m_out, d_n_out, d_news);
    }
    C(cudaPeekAtLastError());

    std::cout << std::format("cudaDeviceSynchronize()\n");
    C(cudaDeviceSynchronize());
    C(cudaMemcpy(&h_n_out, d_n_out, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << std::format("sort()\n");
    std::sort(m_out, m_out + h_n_out);
    std::cout << std::format("verify()\n");
    verify(m_out, h_n_out);

    C(cudaFree(d_news));
    C(cudaFree(m_out));
    C(cudaFree(d_n_out));
}
