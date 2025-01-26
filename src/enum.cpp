#include "Shape.hpp"

#include <atomic>
#include <bit>
#include <unordered_set>
#include <mutex>
#include <iostream>
#include <thread>
#include <ranges>

#define L 8

struct bin_t {
    std::mutex mtx;
    std::unordered_set<Shape<L>> set;
};

static std::array<bin_t, L + 1> bins;
static std::atomic_size_t found;
static uint64_t report;

void worker(uint64_t range, uint64_t mul) {
    for (auto i = 0ull; i < range; i++) {
        auto v = i + mul * range;
        auto n = std::popcount(v);
        if (mul == 0 && i % report == 0)
            std::print(std::cerr, "{}/{} = {:02f}% {} found, n={}\n",
                    i, range, static_cast<double>(i) / range,
                    found.load(std::memory_order_relaxed), n);
        if (n > Shape<L>::LEN)
            continue;
        if (!Shape<L>{ v }.connected())
            continue;

        found.fetch_add(1, std::memory_order_relaxed);
        auto &bin = bins[n];
        std::lock_guard lock{ bin.mtx };
        bin.set.insert(Shape<L>{ v }.canonical_form());
    }
}

int main(int argc, char *argv[]) {
    report = std::stoull(argv[1]);
    {
        std::vector<std::jthread> threads;
        for (auto i = 0; i < 128; i++)
            threads.emplace_back(worker, 1ull << (std::stoi(argv[2]) - 7), i);
    }

    for (auto &&[bin, i] : std::views::zip(bins, std::views::iota(0zu))) {
        if (!i) continue;

        std::print(std::cout, "uint64_t known_shapes_{}[]{{ ", i);
        for (auto sh : bin.set)
            std::cout << sh.get_value() << ", ";
        std::print(std::cout, "}};\n");
    }
}
