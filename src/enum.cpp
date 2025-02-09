#include "Shape.hpp"

#include <atomic>
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <unordered_set>
#include <set>
#include <mutex>
#include <iostream>
#include <thread>
#include <ranges>

#include "util.hpp"

#define L 8

struct bin_t {
    std::mutex mtx;
    std::unordered_set<Shape<L>> set;
};

static std::array<bin_t, L + 1> bins;
static std::atomic_size_t found;
static uint64_t report;
static bool sym_C;

void worker(uint64_t range, uint64_t mul) {
    for (auto i = 0ull; i < range; i++) {
        auto v = i + mul * range;
        auto n = std::popcount(v);
        if (mul == 0 && i % report == 0)
            std::print(std::cerr, "{}/{} = {:02f}% {} found, n={}\n",
                    i, range, 100.0 * i / range,
                    found.load(std::memory_order_relaxed), n);
        if (n > Shape<L>::LEN)
            continue;
        if (!Shape<L>{ v }.connected())
            continue;

        found.fetch_add(1, std::memory_order_relaxed);
        auto &bin = bins[n];
        std::lock_guard lock{ bin.mtx };
        auto sh = Shape<L>{ v }.canonical_form();
        if (sh != Shape<L>{ v })
            continue;
        if (!bin.set.insert(sh).second)
            std::abort();
        if (!sym_C)
            continue;
        auto flip = sh.canonical_form(0b10010110u);
        if (flip == sh)
            continue;
        if (!bin.set.insert(flip).second)
            std::abort();
    }
}

int main(int argc, char *argv[]) {
    sym_C = ::getenv("C") && *::getenv("C");
    auto prefix = sym_C ? "C_" : "";
    report = std::stoull(argv[1]);
    {
        std::vector<std::jthread> threads;
        for (auto i = 0; i < 128; i++)
            threads.emplace_back(worker, 1ull << (std::stoi(argv[2]) - 7), i);
    }

    std::print(std::cout, "#include \"known.hpp\"\n\n");
    for (auto &&[bin, i] : std::views::zip(bins, std::views::iota(0zu))) {
        if (!i) continue;

        std::print(std::cout, "static const uint64_t s{}[]{{ ", i);
        auto values = bin.set | std::views::transform(&Shape<L>::get_value)
            | std::ranges::to<std::vector>();
        std::ranges::sort(values);
        for (auto v : values)
            std::cout << v << ", ";
        std::print(std::cout, "}};\n");
        std::print(std::cerr, "[{}] => {}\n", i, bin.set.size());
    }

    std::print(std::cout, "\nconst uint64_t * const known_{}shapes[]{{ nullptr, ", prefix);
    for (auto &&[bin, i] : std::views::zip(bins, std::views::iota(0zu)))
        if (i)
            std::print(std::cout, "s{}, ", i);
    std::print(std::cout, "}};\n");

    std::print(std::cout, "\nconst size_t shapes_{}count[]{{ ", prefix);
    for (auto &&[bin, i] : std::views::zip(bins, std::views::iota(0zu))) {
        std::print(std::cout, "{}, ", bin.set.size());
    }
    std::print(std::cout, "}};\n");
}
