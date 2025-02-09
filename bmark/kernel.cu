#include <vector>

#include <curand.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <cstdio>
#include <csignal>
#include <iomanip>
#include <unistd.h>
#include <readline/readline.h>
#include <readline/history.h>

#include <format>
#include <iostream>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>

#include "../src/frow.h"
#include "../src/kernel.h"
#include "../src/naming.hpp"
#include "../src/known.hpp"
#include "../src/record.cuh"
#include "../src/util.cuh"
#include "../src/util.hpp"
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

static KSizing ks;

int default_blocks(const std::string &ty, int threads, uint64_t n_cfgs) {
    if (ty == "legacy")
        return (n_cfgs * ks.f0Lsz * ks.f0Rsz + threads - 1) / threads;
    if (ty == "CR")
        return n_cfgs * ks.f0Lsz * ((ks.f0Rsz + threads - 1) / threads);
    if (ty == "CL")
        return n_cfgs * ks.f0Rsz * ((ks.f0Lsz + threads - 1) / threads);
    return (n_cfgs + threads - 1) / threads;
}

int default_shmem(int threads) {
    if (threads <= 96)
        return 7168;
    else if (threads <= 128)
        return 11776;
    else if (threads <= 256)
        return 15872;
    else if (threads <= 384)
        return 24576;
    else if (threads <= 512)
        return 32768;
    else if (threads <= 768)
        return 50176;
    else
        return 101376;
}

static bool running = false;

using namespace std::string_literals;
// ChatGPT generated code, fixed 114514 bugs {{{
static const std::vector<std::string> THREADS{"32", "64", "96", "128", "192", "256", "384", "512", "768", "1024"};

// Readline completion function
char *command_generator(const char *text, int state) {
    static std::vector<std::string> matches;
    static size_t index;

    if (state == 0) {
        matches.clear();
        index = 0;

        std::vector<std::string> tokens;
        boost::split(tokens, rl_line_buffer, boost::is_any_of(" "), boost::token_compress_on);

        // Determine position in the input
        size_t pos = tokens.size();
        bool is_lr = !tokens.empty() && (tokens[0] == "L" || tokens[0] == "R");
        bool is_co = !tokens.empty() && (tokens[0] == "CL" || tokens[0] == "CR");
        bool is_legacy = !tokens.empty() && tokens[0] == "legacy";

        if (pos == 1) {
            if ("legacy"s.starts_with(text)) matches.push_back("legacy");
            if ("list"s.starts_with(text)) matches.push_back("list");
            if ("L"s.starts_with(text)) matches.push_back("L");
            if ("R"s.starts_with(text)) matches.push_back("R");
            if ("CL"s.starts_with(text)) matches.push_back("CL");
            if ("CR"s.starts_with(text)) matches.push_back("CR");
        } else if (pos == 2 && (is_lr || is_co || is_legacy)) {
            // Complete <threads>
            for (const std::string &thread : THREADS) {
                if (thread.starts_with(text))
                    matches.push_back(thread);
            }
        } else if (pos == 3 && (is_lr || is_co || is_legacy)) {
            // Complete <blocks> (computed from `default_blocks`)
            if ("auto"s.starts_with(text)) matches.push_back("auto");
            int threads = std::stoi(tokens[1]);
            int blocks = default_blocks(tokens[0], threads, ks.n_cfgs);
            matches.push_back(std::to_string(blocks));
        } else if (pos == 4 && is_lr) {
            // Complete <shmem> (computed from `default_shmem`)
            if ("auto"s.starts_with(text)) matches.push_back("auto");
            int threads = std::stoi(tokens[1]);
            int shmem = default_shmem(threads);
            matches.push_back(std::to_string(shmem));
        }
    }

    // Return next match
    if (index < matches.size()) {
        return strdup(matches[index++].c_str());
    }
    return nullptr;
}

// Readline wrapper function
char **custom_completer(const char *text, int , int ) {
    return rl_completion_matches(text, command_generator);
}

void handle_sigint(int sig) {
    if (running)
        exit(1);
    // Clear the current input line when Ctrl-C is pressed
    printf("\n");  // Move to a new line
    rl_on_new_line();  // Reset readline's internal state
    rl_replace_line("", 0);  // Clear the input buffer
    rl_redisplay();  // Refresh the prompt
}

// }}} ChatGPT generated code

int main(int argc, char *argv[]) {
    rl_attempted_completion_function = custom_completer;
    rl_variable_bind("show-all-if-ambiguous", "on");
    struct sigaction sa;
    sa.sa_handler = handle_sigint;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);

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
    compute_frow_on_cpu(n_pars >= 0);

    auto szid = std::min(height - 1, 5u);
    auto fanoutL = h_frowInfoL[(ea >> 0) & 0xfu].sz[szid];
    auto fanoutR = h_frowInfoR[(ea >> 4) & 0xfu].sz[szid];

    prepare_kernels();

    cudaStream_t stream;
    C(cudaStreamCreate(&stream));

    // for unknown reason, accessing d_frowDataX gives invalid memory access
    // transfer_frow_to_gpu();
    frow_info_d f0L, f0R;
    std::cout << std::format("copy f0L({}), f0R({}) at szid={}\n", fanoutL, fanoutR, szid);
    C(cudaMallocAsync(&f0L.data32, fanoutL*sizeof(frow32_t), stream));
    C(cudaMallocAsync(&f0R.data32, fanoutR*sizeof(frow32_t), stream));
    C(cudaMemcpyAsync(f0L.data32, h_frowInfoL[(ea >> 0) & 0xfu].data32,
                fanoutL*sizeof(frow32_t), cudaMemcpyHostToDevice, stream));
    C(cudaMemcpyAsync(f0R.data32, h_frowInfoR[(ea >> 4) & 0xfu].data32,
                fanoutR*sizeof(frow32_t), cudaMemcpyHostToDevice, stream));
    C(cudaMallocAsync(&f0L.dataL, fanoutL*sizeof(uint32_t), stream));
    C(cudaMallocAsync(&f0R.dataL, fanoutR*sizeof(uint32_t), stream));
    C(cudaMemcpyAsync(f0L.dataL, h_frowInfoL[(ea >> 0) & 0xfu].dataL,
                fanoutL*sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    C(cudaMemcpyAsync(f0R.dataL, h_frowInfoR[(ea >> 4) & 0xfu].dataL,
                fanoutR*sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    C(cudaMallocAsync(&f0L.dataH, fanoutL*sizeof(uint32_t), stream));
    C(cudaMallocAsync(&f0R.dataH, fanoutR*sizeof(uint32_t), stream));
    C(cudaMemcpyAsync(f0L.dataH, h_frowInfoL[(ea >> 0) & 0xfu].dataH,
                fanoutL*sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    C(cudaMemcpyAsync(f0R.dataH, h_frowInfoR[(ea >> 4) & 0xfu].dataH,
                fanoutR*sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    C(cudaMallocAsync(&f0L.data0123, fanoutL*sizeof(uint32_t), stream));
    C(cudaMallocAsync(&f0R.data0123, fanoutR*sizeof(uint32_t), stream));
    C(cudaMemcpyAsync(f0L.data0123, h_frowInfoL[(ea >> 0) & 0xfu].data0123,
                fanoutL*sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    C(cudaMemcpyAsync(f0R.data0123, h_frowInfoR[(ea >> 4) & 0xfu].data0123,
                fanoutR*sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

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
    unsigned long long *perf;
    C(cudaMallocAsync(&perf, 4 * sizeof(long long), stream));

    cudaDeviceProp prop;
    C(cudaGetDeviceProperties(&prop, 0));

    boost::iostreams::stream<boost::iostreams::file_descriptor_sink> csv(3,
            boost::iostreams::close_handle);
    csv << "n_cfgs,f0Lsz,f0Rsz,reverse,blocks,threads,shmem_len,fom,height,ea,n_outs,clockRate,oc,e,perf_lr,perf_n,perf_tile,compI,ex\n";

    auto launch = [&](const KParams &kp) {
        running = true;
        C(cudaMemsetAsync(n_outs, 0, sizeof(unsigned long long), stream));
        C(cudaMemsetAsync(perf, 0, 4 * sizeof(unsigned long long), stream));
        std::cout << kp.to_string(false) << " => ";
        std::cout.flush();
        std::cout.flush();

        KParamsFull kpf{ kp, height,
            ring_buffer, n_outs, N_CHUNKS,
            nullptr, nullptr, cfgs, ea,
            f0L, // d_frowDataL[0][ea >> 0 & 0xfu],
            f0R, // d_frowDataR[0][ea >> 4 & 0xfu],
            perf };
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
        unsigned long long outs;
        C(cudaMemcpyAsync(&outs, n_outs, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
        unsigned long long perfs[4];
        C(cudaMemcpyAsync(&perfs, perf, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
        C(cudaStreamSynchronize(stream));
        auto oc = std::min(16u, 1536u / kpf.threads) * 84; // max blocks per device
        auto e = ((kpf.blocks + oc - 1) / oc);
        // auto tpg = kpb.blocks * kpb.threads;
        // auto iterations = (kpf.n_cfgs + tpg - 1) / tpg;
        auto rt = (kpf.n_cfgs + kpf.threads - 1) / kpf.threads * kpf.threads
            * 1000.0 * prop.clockRate / e;
        auto perf_lr = perfs[0];
        auto perf_n = perfs[1];
        auto perf_tile = perfs[2];
        auto perf_comp = perfs[3];
        auto ex = 1000 * (perf_lr + perf_n + perf_tile) / rt / ms;
        std::cout << std::format("{} / {:.01f}ms = ({:>7}+{:>7}+{:>7})*{} I{:.02f}% E{:.02f}% raw({:>7}+{:>7}+{:>7})\n",
                outs, ms,
                display(perf_lr / rt / ex / e), display(perf_n / rt / ex / e), display(perf_tile / rt / ex / e),
                e,
                100.0 * perf_comp / perf_tile, 100.0 * ex,
                display(perf_lr / rt), display(perf_n / rt), display(perf_tile / rt));
        running = false;
        C(cudaEventDestroy(start));
        C(cudaEventDestroy(stop));
        csv << std::setprecision(17);
        csv << kpf.n_cfgs << ",";
        csv << kpf.f0Lsz << ",";
        csv << kpf.f0Rsz << ",";
        switch (kpf.ty) {
            case KKind::Legacy: csv << "legacy"; break;
            case KKind::CoalescedR: csv << "CR"; break;
            case KKind::CoalescedL: csv << "CL"; break;
            case KKind::TiledStandard: csv << "R"; break;
            case KKind::TiledReversed: csv << "L"; break;
        }
        csv << kpf.blocks << ",";
        csv << kpf.threads << ",";
        csv << kpf.shmem_len << ",";
        csv << kpf.fom() << ",";
        csv << kpf.height << ",";
        csv << (int)kpf.ea << ",";
        csv << outs << ",";
        csv << prop.clockRate << ",";
        csv << oc << ",";
        csv << e << ",";
        csv << 1e6 * perf_lr / rt / ex / e << ",";
        csv << 1e6 * perf_n / rt / ex / e << ",";
        csv << perf_tile / rt / ex / e << ",";
        csv << 1.0 * perf_comp / perf_tile << ",";
        csv << ex << std::endl;
    };

    ks = KSizing{ n_cfgs, fanoutL, fanoutR };
    KParams kp;
    std::cout << R"(<COMMAND> ::=)" << "\n";
    std::cout << R"(    | "list")" << "\n";
    std::cout << R"(    | "legacy"    <threads> [(<blocks>|"auto") [<n_cfg>])" << "\n";
    std::cout << R"(    | ("CL"|"CR") <threads> [(<blocks>|"auto") [<n_cfg>])" << "\n";
    std::cout << R"(    | ("L"|"R")   <threads> [(<blocks>|"auto") [(<shmem>|"auto") [<n_cfg>]])" << "\n";
    while (n_pars == -1) {
        auto input = readline("> ");
        if (input == nullptr)
            break;

        std::vector<std::string> tokens;
        std::string line{ input };
        free(input);
        boost::split(tokens, line, boost::is_any_of(" "), boost::token_compress_on);
        if (!tokens.empty() && tokens.back().empty())
            tokens.pop_back();
        if (tokens.empty())
            continue;
        if (tokens[0] == "list") {
            auto pars = ks.optimize();
            for (auto i = 0zu; i < pars.size() && i < 20zu; i++)
                pars[i].fom(true);
            continue;
        }
        if (tokens.size() < 2 || tokens.size() > 5) {
            std::cout << "invalid <KParams>\n";
            continue;
        }
        if (tokens[0] == "legacy") {
            kp = KParams{ ks, KKind::Legacy };
            kp.threads = std::stoull(tokens[1]);
            if (tokens.size() >= 4)
                kp.n_cfgs = std::stoull(tokens[3]);
            if (tokens.size() < 3 || tokens[2] == "auto")
                kp.blocks = default_blocks(tokens[0], kp.threads, kp.n_cfgs);
            else
                kp.blocks = std::stoull(tokens[2]);
        } else if (tokens[0] == "CL" || tokens[0] == "CR") {
            kp = KParams{ ks, KKind::CoalescedR };
            if (tokens[0] == "CL")
                kp.ty = KKind::CoalescedR;
            kp.threads = std::stoull(tokens[1]);
            if (tokens.size() >= 4)
                kp.n_cfgs = std::stoull(tokens[3]);
            if (tokens.size() < 3 || tokens[2] == "auto")
                kp.blocks = default_blocks(tokens[0], kp.threads, kp.n_cfgs);
            else
                kp.blocks = std::stoull(tokens[2]);
        } else if (tokens[0] == "L" || tokens[0] == "R") {
            kp = KParams{ ks, KKind::TiledStandard };
            if (tokens[0] == "L")
                kp.ty = KKind::TiledReversed;
            kp.threads = std::stoull(tokens[1]);
            if (tokens.size() >= 5)
                kp.n_cfgs = std::stoull(tokens[4]);
            if (tokens.size() < 3 || tokens[2] == "auto") {
                kp.blocks = default_blocks(tokens[0], kp.threads, kp.n_cfgs);
            } else {
                kp.blocks = std::stoull(tokens[2]);
            }
            if (tokens.size() < 4 || tokens[3] == "auto") {
                kp.shmem_len = default_shmem(kp.threads) / sizeof(frow32_t);
            } else {
                kp.shmem_len = std::stoll(tokens[3]) / sizeof(frow32_t);
            }
        } else {
            continue;
        }
        add_history(line.c_str());
        kp.fom(true);
        launch(kp);
    }

    if (n_pars >= 0) {
        C(cuProfilerStart());
        auto pars = ks.optimize();
        if (pars.size() > n_pars)
            pars.erase(pars.begin() + n_pars, pars.end());
        for (auto it = pars.rbegin(); it != pars.rend(); it++)
            it->fom(true);
        for (auto &res : pars) {
            launch(res);
        }
    }

    C(cudaStreamDestroy(stream));
    C(cuProfilerStop());
}
