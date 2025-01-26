#include <algorithm>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <ranges>
#include <print>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "Shape.hpp"
#include "Piece.inl"

void cover(std::unordered_map<Shape, size_t> &map, const std::vector<Piece> &lib, Shape board, bool append) {
    if (lib.empty()) return;
    size_t solutions{};
    [&](this auto &&self, auto it, size_t cnt, Shape open_tiles) {
        while (!cnt && ++it != lib.end())
            cnt = it->count;
        if (it == lib.end()) {
            solutions++;
            if (solutions % 1000000 == 0)
                std::print("at {}\n", solutions);
            auto map_it = map.find(open_tiles);
            if (map_it == map.end()) {
                if (append)
                    map.emplace(open_tiles, 1);
            } else {
                map_it->second++;
            }
            return;
        }
        auto &piece = *it;
        cnt--;
        if (piece.canonical.size() * open_tiles.size() >= Shape::LEN * Shape::LEN) {
            piece.cover([&](Shape placed, size_t, coords_t) {
                if (open_tiles >= placed)
                    self(it, cnt, open_tiles - placed);
                return false;
            });
        } else {
            std::unordered_set<Shape> seen;
            for (auto coords : board) {
                piece.cover(coords, [&](Shape placed, size_t, coords_t) {
                    if (!(open_tiles >= placed)) return false;
                    if (!seen.insert(placed).second) return false;
                    self(it, cnt, open_tiles - placed);
                    return false;
                });
            }
        }
    }(lib.begin(), lib.front().count, board);
}

int main() {
    std::vector<Shape::shape_t> sh{
        0b11110000'10000000u,          // L
        0b11100000'00110000u,          // Z
        0b01100000'01000000'11000000u, // S
        0b10000000'10000000'11100000u, // L_
        0b11100000'11100000u,          // rect
        // 0b11100000'11000000u,          // rect-
        // 0b11100000'10100000u,          // C
        // 0b11110000'00100000u,          // |-
    };
    auto pieces = sh
        | std::views::transform([](Shape::shape_t sh){ return Piece{ Shape{ sh } }; })
        | std::ranges::to<std::vector>();
    Shape board{ 0b11111100'11111100'11111110'11111110'11111110'11111110'11100000ull };
    board = board.transform<false, true, true>(true);

    std::unordered_map<Shape, size_t> map;
    map.reserve(1zu << 24);
    // for (auto month = 1; month <= 12; month++) {
    //     for (auto day = 1; day <= 31; day++) {
    //         auto m = month;
    //         auto d = day;
    //         auto b = board
    //             .clear((m - 1) / 6, (m - 1) % 6)
    //             .clear((d - 1) / 7 + 2, (d - 1) % 7);
    //         map.emplace(b, 0);
    //     }
    // }

    cover(map, pieces, board, true);
    auto [min, max] = std::ranges::minmax(std::views::elements<1>(map));
    std::print("min={} max={} sz={} lf={}\n", min, max, map.size(), map.load_factor());
}
