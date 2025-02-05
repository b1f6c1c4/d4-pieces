#include "../src/sorter.hpp"

#include <atomic>
#include <random>
#include <mutex>
#include <mimalloc-new-delete.h>
#include <print>
#include <iostream>

#include "../src/util.hpp"

#define BOOST_THREAD_VERSION 5
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>

struct CudaSearcher {
    R *d;
    Rg<R> write_solution(unsigned, size_t sz);
};
Rg<R> CudaSearcher::write_solution(unsigned, size_t sz) {
    std::atomic_ref atm{ d };
    return Rg<R>{ atm.fetch_add(sz, std::memory_order_relaxed), sz };
}

int main(int argc, char *argv[]) {
    auto sz = std::stoll(argv[1]);
    auto chunk = std::stoll(argv[2]) * 1048576ull / sizeof(RX); // MiB
    auto amplify = std::stoll(argv[3]);
    std::vector<Rg<RX>> data;
    std::bernoulli_distribution dist{ 1 / 1.3 };
    auto dest = malloc(sz * chunk * sizeof(R));
    std::print("generating {} * {}B = {}B test data\n",
            sz, display(chunk * sizeof(RX)), display(sz * chunk * sizeof(RX)));
    std::mutex mtx;
    boost::basic_thread_pool pool;
    for (auto c = 0ull; c < sz; c++)
        boost::async(pool, [&](uint64_t seed) {
            std::mt19937_64 rnd{};
            rnd.seed(seed);
            Rg<RX> r{
                reinterpret_cast<RX *>(std::aligned_alloc(4096, chunk * sizeof(RX))),
                chunk };
            for (auto i = 0; i < chunk; i++) {
                if (!dist(rnd) && i) {
                    auto lucky = rnd() % i;
                    r.ptr[i] = r.ptr[lucky];
                    continue;
                }
                r.ptr[i].xaL = rnd();
                r.ptr[i].xaH = rnd() & 0x0000ffffu;
                r.ptr[i].xaH |= (rnd() % 17u) << 24;
                r.ptr[i].ex0 = rnd();
                r.ptr[i].ex1 = rnd();
                r.ptr[i].ex2 = rnd();
                r.ptr[i].ea = rnd();
            }
            std::lock_guard lock{ mtx };
            data.push_back(r);
        }, c);
    pool.close();
    pool.join();

    std::print("launch sorter\n");
    CudaSearcher cs{ reinterpret_cast<R *>(dest) };
    Sorter sorter{ cs };
    std::print("dispatching data\n");
    for (auto r : data)
        for (auto i = 0; i < amplify; i++)
            sorter.push(r, 6);
    std::print("Sorter::join\n");
    auto t1 = std::chrono::steady_clock::now();
    sorter.join();
    auto t2 = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    if (us < 1000)
        std::cout << std::format("  => completed in {}us\n", us);
    else if (us < 1000000)
        std::cout << std::format("  => completed in {:.2f}ms\n", us / 1e3);
    else
        std::cout << std::format("  => completed in {:.2f}s\n", us / 1e6);
}
