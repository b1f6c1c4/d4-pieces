#include <vector>

#include <curand.h>

#include "../src/frow.h"
#include "../src/kernel.h"

#include <format>
#include <iostream>

#include "../src/naming.hpp"
#include "../src/known.hpp"

#include "../src/util.cuh"

static inline void chk_impl(curandStatus_t code, const char *file, int line) {
    if (code != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error{
            std::format("curand: {} @ {}:{}\n", (int)code, file, line) };
    }
}

// defined in frow.cpp
extern std::optional<Naming> g_nme;
extern unsigned g_sym;

#define N_CHUNKS 47
__device__  RX                 ring_buffer[N_CHUNKS*CYC_CHUNK/sizeof(RX)];
__device__  unsigned long long n_outs;
__managed__ unsigned long long nrc, nwc;
__managed__ R *cfgs;

int main(int argc, char *argv[]) {
    if (argc != 9) {
        std::cout << std::format(
                "Usage: {} <min_m> <max_m> <min_n> <max_n> <board_n> <ea> <height> <n_cfgs>\n",
                argv[0]);
        return 1;
    }
    auto sym_C = ::getenv("C") && *::getenv("C");
    g_sym = sym_C ? 0b01101001u : 0b11111111u;
    auto min_m = std::atoi(argv[1]);
    auto max_m = std::atoi(argv[2]);
    auto min_n = std::atoi(argv[3]);
    auto max_n = std::atoi(argv[4]);
    auto board_n = std::atoi(argv[5]);
    auto ea = (uint8_t)std::strtol(argv[6], nullptr, 16);
    auto height = (unsigned)std::atoi(argv[7]);
    auto n_cfgs = (uint64_t)std::atoll(argv[8]);
    g_nme.emplace(
        (uint64_t)min_m, (uint64_t)max_m,
        (uint64_t)min_n, (uint64_t)max_n,
        board_n,
        sym_C ? known_C_shapes : known_shapes,
        sym_C ? shapes_C_count : shapes_count);

    compute_frow_on_cpu();

    auto szid = std::min(height - 1, 5u);
    auto fanoutL = h_frowInfoL[(ea >> 0) & 0xfu].sz[szid];
    auto fanoutR = h_frowInfoR[(ea >> 4) & 0xfu].sz[szid];

    prepare_kernels();

    cudaStream_t stream;
    C(cudaStreamCreate(&stream));

    frow_t *f0L, *f0R;
    std::cout << std::format("copy f0L({}), f0R({}) at szid={}\n", fanoutL, fanoutR, szid);
    C(cudaMallocAsync(&f0L, fanoutL*sizeof(frow_t), stream));
    C(cudaMallocAsync(&f0R, fanoutR*sizeof(frow_t), stream));
    C(cudaMemcpyAsync(f0L, h_frowInfoL[(ea >> 0) & 0xfu].data,
                fanoutL*sizeof(frow_t), cudaMemcpyHostToDevice, stream));
    C(cudaMemcpyAsync(f0R, h_frowInfoR[(ea >> 0) & 0xfu].data,
                fanoutR*sizeof(frow_t), cudaMemcpyHostToDevice, stream));

    std::cout << std::format("allocate {} cfgs\n", n_cfgs);
    C(cudaMallocManaged(&cfgs, n_cfgs*sizeof(R)));
    std::cout << std::format("randomize {} cfgs\n", n_cfgs);
    curandGenerator_t gen;
    C(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    C(curandSetPseudoRandomGeneratorSeed(gen, 23336666));
    C(curandGenerateUniformDouble(gen,
                reinterpret_cast<double *>(cfgs), n_cfgs*sizeof(R)/sizeof(double)));

    auto pars = KSizing{ n_cfgs, fanoutL, fanoutR }.optimize();
    pars.erase(pars.begin() + 10, pars.end());
    for (auto res : pars) {
        unsigned long long tmp{};
        C(cudaMemcpyToSymbolAsync(n_outs, &tmp, 0,
                    sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
        C(cudaMemcpyToSymbolAsync(nrc, &tmp, 0,
                    sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
        C(cudaMemcpyToSymbolAsync(nwc, &tmp, 0,
                    sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
        if (res.shmem_len) {
            std::cout << std::format("<<<{:9},{:5},{:5}B>>>[{}]/{:.02e} => ",
                    res.blocks, res.threads, res.shmem_len * sizeof(frow_t),
                    res.reverse ? "R" : "L",
                    res.fom());
        } else {
            std::cout << std::format("<<<{:9},{:5}>>>  [legacy]/{:.02e} => ",
                    res.blocks, res.threads,
                    res.fom());
        }
        std::cout.flush();

        KParamsFull kpf{ res, height,
            ring_buffer, &n_outs, N_CHUNKS,
            &nrc, &nwc, cfgs, ea, f0L, f0R };
        cudaEvent_t start, stop;
        C(cudaEventCreate(&start));
        C(cudaEventCreate(&stop));
        C(cudaEventRecord(start));
        kpf.launch(stream);
        C(cudaPeekAtLastError());
        C(cudaEventRecord(stop));
        C(cudaEventSynchronize(stop));
        float ms;
        C(cudaEventElapsedTime(&ms, start, stop));
        C(cudaMemcpyFromSymbolAsync(&tmp, n_outs, 0,
                    sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
        C(cudaStreamSynchronize(stream));
        std::cout << std::format("{:.08f}ms, n_out={}\n", ms, n_outs);
        C(cudaEventDestroy(start));
        C(cudaEventDestroy(stop));
    }

    C(cudaStreamDestroy(stream));
}
