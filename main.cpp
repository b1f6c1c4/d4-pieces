#include <filesystem>
#include <fmt/ranges.h>
#include <fstream>
#include <ranges>
#include <sstream>
#include <thread>
#include <unordered_map>

#include "Shape.hpp"
#include "Piece.hpp"

int main() {
    Library library{
        0b11110000'10000000u,          // L
        0b11100000'00110000u,          // Z
        0b01100000'01000000'11000000u, // S
        0b10000000'10000000'11100000u, // L_
        0b11100000'11100000u,          // rect
        0b11100000'11000000u,          // rect-
        0b11100000'10100000u,          // C
        0b11110000'00100000u,          // |-
    };
    Shape board{ 0b11111100'11111100'11111110'11111110'11111110'11111110'11100000ull };
    board = board.transform<false, true, true>(true);

    std::vector<std::jthread> threads;
    for (auto month = 1; month <= 12; month++) {
        for (auto day = 1; day <= 31; day++)
            threads.emplace_back([&](int m, int d){
                auto solutions = library.solve(board
                        .clear((m - 1) / 6, (m - 1) % 6)
                        .clear((d - 1) / 7 + 2, (d - 1) % 7));
                fmt::println("{:02}/{:02} => {} solutions", m, d, solutions.size());
            }, month, day);
    }
    return 0;
}
