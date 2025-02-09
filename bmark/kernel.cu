#include <vector>

#include <curand.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>

#include <format>
#include <iostream>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include "../src/frow.h"
#include "../src/kernel.h"
#include "../src/naming.hpp"
#include "../src/known.hpp"
#include "../src/record.cuh"
#include "../src/util.cuh"
#include "../src/sn.cuh"

static inline void chk_impl(curandStatus_t code, const char *file, int line) {
    if (code != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error{
            std::format("curand: {} @ {}:{}\n", (int)code, file, line) };
    }
}

// defined in frow.cpp
extern std::optional<Naming> g_nme;
extern unsigned g_sym;

// #define N_CHUNKS 45
#define N_CHUNKS 2
__managed__ R *cfgs;

template <unsigned H>
__launch_bounds__(768, 2)
__global__ void fix_cfgs(unsigned long long n_cfgs) {
    auto idx = threadIdx.x + (uint64_t)blockDim.x * blockIdx.x;
    if (idx >= n_cfgs) return;
    auto cfg = parse_R<H>(cfgs[idx], 0x00);
    cfg.empty_area = ~0ull;
    switch (H) {
        case 8: cfg.nm_cnt = 0u; break;
        case 7: cfg.nm_cnt = (cfg.nm_cnt % 8u); break;
        case 6: cfg.nm_cnt = (cfg.nm_cnt % 14u); break;
        case 5: cfg.nm_cnt = (cfg.nm_cnt % 15u); break;
        default: cfg.nm_cnt = (cfg.nm_cnt % 16u); break;
    }
    d_sn(cfg.nm_cnt, cfg.ex);
    auto *nm = reinterpret_cast<uint8_t *>(cfg.ex);
    if (cfg.nm_cnt && nm[0] != 0xff) {
        auto ub = 0u; // [0,ub] are unique
        for (auto i = 1u; i < cfg.nm_cnt; i++) {
            if (nm[i] == 0xff)
                break;
            if (nm[i] != nm[ub])
                nm[++ub] = nm[i];
        }
        cfg.nm_cnt = ub;
    }
    for (auto i = cfg.nm_cnt; i < 16u; i++)
        nm[i] = 0xff;
    cfgs[idx] = assemble_R<H>(cfg);
}

void launch_fix_cfgs(unsigned H, unsigned long long n_cfgs, cudaStream_t s) {
    switch (H) {
        case 7: fix_cfgs<7><<<(n_cfgs + 768 - 1) / 768, 768, 0, s>>>(n_cfgs); break;
        case 6: fix_cfgs<6><<<(n_cfgs + 768 - 1) / 768, 768, 0, s>>>(n_cfgs); break;
        case 5: fix_cfgs<5><<<(n_cfgs + 768 - 1) / 768, 768, 0, s>>>(n_cfgs); break;
        case 4: fix_cfgs<4><<<(n_cfgs + 768 - 1) / 768, 768, 0, s>>>(n_cfgs); break;
        case 3: fix_cfgs<3><<<(n_cfgs + 768 - 1) / 768, 768, 0, s>>>(n_cfgs); break;
        case 2: fix_cfgs<2><<<(n_cfgs + 768 - 1) / 768, 768, 0, s>>>(n_cfgs); break;
        case 1: fix_cfgs<1><<<(n_cfgs + 768 - 1) / 768, 768, 0, s>>>(n_cfgs); break;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 9 && argc != 10) {
        std::cout << std::format(
                "Usage: {} <min_m> <max_m> <min_n> <max_n> <board_n> <ea> <height> <n_cfgs> [<n_pars>]\n",
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
    auto n_pars{ -1 };
    if (argc == 10)
        n_pars = std::atoi(argv[9]);
    g_nme.emplace(
            (uint64_t)min_m, (uint64_t)max_m,
            (uint64_t)min_n, (uint64_t)max_n,
            board_n,
            sym_C ? known_C_shapes : known_shapes,
            sym_C ? shapes_C_count : shapes_count);

    if (n_pars >= 0)
        show_gpu_devices();
    compute_frow_on_cpu();

    auto szid = std::min(height - 1, 5u);
    auto fanoutL = h_frowInfoL[(ea >> 0) & 0xfu].sz[szid];
    auto fanoutR = h_frowInfoR[(ea >> 4) & 0xfu].sz[szid];

    prepare_kernels();

    cudaStream_t stream;
    C(cudaStreamCreate(&stream));

    frow32_t *f0L, *f0R;
    std::cout << std::format("copy f0L({}), f0R({}) at szid={}\n", fanoutL, fanoutR, szid);
    C(cudaMallocAsync(&f0L, fanoutL*sizeof(frow32_t), stream));
    C(cudaMallocAsync(&f0R, fanoutR*sizeof(frow32_t), stream));
    C(cudaMemcpyAsync(f0L, h_frowInfoL[(ea >> 0) & 0xfu].data32,
                fanoutL*sizeof(frow32_t), cudaMemcpyHostToDevice, stream));
    C(cudaMemcpyAsync(f0R, h_frowInfoR[(ea >> 0) & 0xfu].data32,
                fanoutR*sizeof(frow32_t), cudaMemcpyHostToDevice, stream));

    std::cout << std::format("allocate {} cfgs\n", n_cfgs);
    C(cudaMallocManaged(&cfgs, n_cfgs*sizeof(R)));
    std::cout << std::format("randomize {} cfgs\n", n_cfgs);
    curandGenerator_t gen;
    C(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    C(curandSetPseudoRandomGeneratorSeed(gen, 23336666));
    C(curandGenerateUniformDouble(gen,
                reinterpret_cast<double *>(cfgs), n_cfgs*sizeof(R)/sizeof(double)));
    std::cout << std::format("patch {} cfgs\n", n_cfgs);
    launch_fix_cfgs(height, n_cfgs, stream);
    C(cudaPeekAtLastError());

    RX *ring_buffer;
    // C(cudaMallocManaged(&ring_buffer, N_CHUNKS*CYC_CHUNK*sizeof(RX)));
    C(cudaMallocAsync(&ring_buffer, N_CHUNKS*CYC_CHUNK*sizeof(RX), stream));
    unsigned long long *n_outs;
    C(cudaMallocAsync(&n_outs, sizeof(unsigned long long), stream));

    auto launch = [&](const KParams &kp) {
        unsigned long long tmp{};
        C(cudaMemcpyAsync(n_outs, &tmp,
                    sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
        std::cout << kp.to_string(false) << " => ";
        std::cout.flush();
        std::cout.flush();

        KParamsFull kpf{ kp, height,
            ring_buffer, n_outs, N_CHUNKS,
            nullptr, nullptr, cfgs, ea, f0L, f0R };
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
        C(cudaMemcpyAsync(&tmp, n_outs, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
        C(cudaStreamSynchronize(stream));
        std::cout << std::format("{} / {:.08f}ms\n", tmp, ms);
        C(cudaEventDestroy(start));
        C(cudaEventDestroy(stop));
    };

    KParams kp{ KSizing{ n_cfgs, fanoutL, fanoutR } };
    while (n_pars == -1) {
        std::cout << R"(<KParams> ::=)" << "\n";
        std::cout << R"(    | "legacy" <threads> [<blocks>])" << "\n";
        std::cout << R"(    | ("L"|"R") <threads> [<blocks> [<shmem>]])" << "\n";
        std::cout << "> ";
        std::cout.flush();
        std::vector<std::string> tokens;
        std::string line;
        std::getline(std::cin, line);
        boost::split(tokens, line, boost::is_any_of(" "));
        if (std::cin.eof())
            break;
        if (tokens.empty() || tokens.size() > 4) {
            std::cout << "invalid <KParams>\n";
            continue;
        }
        if (tokens[0] == "legacy") {
            if (tokens.size() < 2) {
                std::cout << "invalid <KParams>\n";
                continue;
            }
            kp = KParams{ KSizing{ n_cfgs, fanoutL, fanoutR } };
            kp.threads = std::stoull(tokens[1]);
            if (tokens.size() < 3)
                kp.blocks = (n_cfgs * fanoutL * fanoutR + kp.threads - 1) / kp.threads;
            else
                kp.blocks = std::stoull(tokens[2]);
        } else if (tokens[0] == "L" || tokens[0] == "R") {
            if (tokens.size() < 2) {
                std::cout << "invalid <KParams>\n";
                continue;
            }
            kp = KParams{ KSizing{ n_cfgs, fanoutL, fanoutR } };
            if (tokens[0] == "L")
                kp.reverse = true;
            kp.threads = std::stoull(tokens[1]);
            if (tokens.size() < 3) {
                kp.blocks = (n_cfgs + kp.threads - 1) / kp.threads;
            } else {
                kp.blocks = std::stoull(tokens[2]);
            }
            if (tokens.size() < 4) {
                if (kp.threads <= 96)
                    kp.shmem_len = 7168 / sizeof(frow32_t);
                else if (kp.threads <= 128)
                    kp.shmem_len = 11776 / sizeof(frow32_t);
                else if (kp.threads <= 256)
                    kp.shmem_len = 15872 / sizeof(frow32_t);
                else if (kp.threads <= 384)
                    kp.shmem_len = 24576 / sizeof(frow32_t);
                else if (kp.threads <= 512)
                    kp.shmem_len = 32768 / sizeof(frow32_t);
                else if (kp.threads <= 768)
                    kp.shmem_len = 50176 / sizeof(frow32_t);
                else
                    kp.shmem_len = 101376 / sizeof(frow32_t);
            } else {
                kp.shmem_len = std::stoll(tokens[3]) / sizeof(frow32_t);
            }
        } else if (tokens[0] == "launch") {
            launch(kp);
            continue;
        }
        std::cout << kp.to_string(true) << "\n";
    }

    if (n_pars >= 0) {
        C(cuProfilerStart());
        auto pars = KSizing{ n_cfgs, fanoutL, fanoutR }.optimize();
        for (auto it = pars.rbegin(); it != pars.rend(); it++)
            (void)it->fom();
        if (pars.size() > n_pars)
            pars.erase(pars.begin() + n_pars, pars.end());
        for (auto &res : pars) {
            launch(res);
        }
    }

    C(cudaStreamDestroy(stream));
    C(cuProfilerStop());
}
