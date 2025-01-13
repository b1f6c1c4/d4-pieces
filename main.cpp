#include <filesystem>
#include <fmt/ranges.h>
#include <fstream>
#include <ranges>
#include <sstream>
#include <thread>
#include <unordered_map>

#include "Shape.hpp"
#include "Piece.hpp"

auto visualize(const std::vector<Shape> &history) {
    std::stringstream output;
    for (auto row = 0u; row < Shape::LEN; row++) {
        for (auto col = 0u; col < Shape::LEN; col++) {
            auto it = std::ranges::find_if(history, [=](Shape sh){ return sh.test(row, col); });
            if (it == history.end())
                output << ' ';
            else
                output << std::distance(history.begin(), it);
        }
        output << '\n';
    }
    return output.str();
}

auto solve(const std::vector<Piece> &library, Shape board) {
    std::vector<std::string> solutions;
    [&](this auto &&self, unsigned avail_pieces, Shape open_tiles, std::vector<Shape> history) {
        if (!open_tiles) {
            solutions.emplace_back(visualize(history));
            return;
        }
        auto pos = open_tiles.front();
        history.emplace_back(0u);
        for (auto pcs = avail_pieces; pcs; pcs -= pcs & -pcs) {
            library[std::countr_zero(pcs)].cover(pos, [&](Shape placed) {
                if (!(open_tiles >= placed)) return;
                history.back() = placed;
                self(avail_pieces - (pcs & -pcs), open_tiles - placed, history);
            });
        }
    }((1u << library.size()) - 1u, board, {});
    return solutions;
}

int main() {
    std::vector<Piece> library{
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
        std::filesystem::create_directories(fmt::format("output/{:02}", month));
        for (auto day = 1; day <= 31; day++)
            threads.emplace_back([&](int m, int d){
                auto solutions = solve(library, board
                        .clear((m - 1) / 6, (m - 1) % 6)
                        .clear((d - 1) / 7 + 2, (d - 1) % 7));
                auto fn = fmt::format("output/{:02}/{:02}.txt", m, d);
                fmt::println("{} => {} solutions", fn, solutions.size());
                std::ofstream fout(fn);
                if (!fout.is_open()) {
                    fmt::println(stderr, "Cannot write\n");
                    return;
                }
                for (auto &sol : solutions)
                    fout << sol << "\n";
                fout.close();
            }, month, day);
    }
    return 0;
}
