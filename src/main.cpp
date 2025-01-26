#include <algorithm>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <ranges>
#include <print>
#include <thread>
#include <unordered_map>

#include "Shape.hpp"
#include "Piece.hpp"

int main() {
    std::vector<Shape::shape_t> sh{
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
        | std::views::transform([](Shape::shape_t sh){ return Piece{ Shape{ sh } }; })
        | std::ranges::to<std::vector>();
    Shape board{ 0b11111100'11111100'11111110'11111110'11111110'11111110'11100000ull };
    board = board.transform<false, true, true>(true);

    std::unordered_map<Shape, size_t> map;
    for (auto month = 1; month <= 12; month++) {
        for (auto day = 1; day <= 31; day++) {
            auto m = month;
            auto d = day;
            auto b = board
                .clear((m - 1) / 6, (m - 1) % 6)
                .clear((d - 1) / 7 + 2, (d - 1) % 7);
            map.emplace(b, 0);
        }
    }

    cover(map, pieces, false);
    auto [min, max] = std::ranges::minmax(std::views::elements<1>(map));
    std::print("min={} max={}\n", min, max);
}
