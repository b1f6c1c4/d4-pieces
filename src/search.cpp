#include <algorithm>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <limits>
#include <ranges>
#include <print>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <unordered_set>

#include "Piece.hpp"
#include "Shape.hpp"
#include "Piece.inl"

template <typename T>
T update(std::atomic<T> &atm, T value, auto &&func) {
    auto o = atm.load(std::memory_order_relaxed);
    auto n = func(o, value);
    while (!atm.compare_exchange_weak(o, n, std::memory_order_relaxed)) {
        n = func(o, value);
        if (n == o)
            break;
    }
    return n;
}

int main() {
    std::vector<Shape<8>::shape_t> sh{
        0b11110000'10000000u,          // L
        0b11100000'00110000u,          // Z
        0b01100000'01000000'11000000u, // S
        0b10000000'10000000'11100000u, // L_
        0b11100000'11100000u,          // rect
        0b11100000'11000000u,          // rect-
        0b11100000'10100000u,          // C
        0b11110000'00100000u,          // |-
    };
    auto pieces = sh
        | std::views::transform([](Shape<8>::shape_t sh){ return Piece<8>{ Shape<8>{ sh } }; })
        | std::ranges::to<std::vector>();
    Shape<8> board{ 0b11111100'11111100'11111110'11111110'11111110'11111110'11100000ull };
    board = board.transform<false, true, true>(true);

    std::atomic<size_t> min = std::numeric_limits<size_t>::max();
    std::atomic<size_t> max = std::numeric_limits<size_t>::min();
    {
        std::vector<std::jthread> threads;
        for (auto month = 1; month <= 12; month++) {
            for (auto day = 1; day <= 31; day++) {
                threads.emplace_back([&](Shape<8> b) {
                    auto c = solve_count(pieces, b);
                    update(min, c, [](size_t a, size_t b){ return std::min(a, b); });
                    update(max, c, [](size_t a, size_t b){ return std::max(a, b); });
                }, board
                    .clear((month - 1) / 6, (month - 1) % 6)
                    .clear((day - 1) / 7 + 2, (day - 1) % 7));
            }
        }
    }

    std::print("min={} max={}\n", min.load(), max.load());
}
