#include "searcher_cuda.h"

#include <linux/kcmp.h>
#include <vector>
#include <memory>
#include <thread>

#include "device.h"
#include "frow.h"
#include "util.hpp"

CudaSearcher::CudaSearcher(uint64_t empty_area)
    : solutions{}, height{ (std::bit_width(empty_area) + 8u - 1u) / 8u } {
    solutions.emplace_back(
        Rg<R>{
            new R[1]{ { (uint32_t)(empty_area >> 8), (uint32_t)(empty_area >> 8 + 32) } },
            1zu,
            RgType::DELETE,
        },
        empty_area & 0xffu);
}

CudaSearcher::~CudaSearcher() {
    free();
}

void CudaSearcher::free() {
    for (auto &r : solutions)
        r.dispose();
}

void CudaSearcher::search_GPU() {
    Sorter sorter{};
    std::vector<std::unique_ptr<Device>> devs;
    std::vector<int> idx;
    for (auto i = 0; i < n_devices; i++) {
        devs.emplace_back(std::make_unique<Device>(i, height, sorter));
        idx.push_back(i);
    }

    std::mutex mtx;
    auto done = false;
    std::condition_variable cv;
    std::jthread monitor{ [&,this] {
        pthread_setname_np(pthread_self(), "monitor");
        auto sep = syscall(SYS_kcmp, getpid(), getpid(), KCMP_FILE, 1, 2) != 0;
#ifndef VERBOSE
        if (!sep)
            return;
#endif
        auto last = 0u;
        using namespace std::chrono_literals;
        std::unique_lock lock{ mtx };
        while (true) {
            cv.wait_for(lock, 100ms, [&]{ return done; });
            if (done)
                break;

            if (sep && last)
                std::print(std::cerr, "\33[{}F", last);
            last = sorter.print_stats();
            for (auto &dev : devs)
                last += dev->print_stats();
            std::print(std::cerr, "\33[37mheight = {}\33[K\33[0m\33[E\33[J\n", height);
            last += 2;
        }
    } };

    while (!solutions.empty()) {
        std::ranges::sort(idx, std::less{}, [&](int i) {
            return devs[i]->get_etc();
        });
        // Device::c is responsible for free up
        devs[idx.front()]->dispatch(solutions.front());
        solutions.pop_front();
    }
    for (auto &dev : devs) dev->close();
    // ensure all compute are finished & results are removed from GPU
    for (auto &dev : devs) dev->wait();
    solutions = sorter.join();
    // stop printing stats before ~Device()
    done = true, cv.notify_one(), monitor.join();
    devs.clear();
    height--;
}

uint64_t CudaSearcher::next_size(unsigned i) const {
    auto szid = min(height - 1, 5);
    auto &w = solutions[i];
    return w.len
        * h_frowInfoL[(w.pos >> 0) & 0b1111u].sz[szid]
        * h_frowInfoR[(w.pos >> 4) & 0b1111u].sz[szid];
}
