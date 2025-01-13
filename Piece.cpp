#include "Piece.hpp"

#include <algorithm>
#include <ranges>
#include <stdexcept>

Piece::Piece(Shape s) : count{ 1 }, canonical{ s } {
    for (auto sh : canonical.transforms(true)) {
        auto duplicate = std::ranges::find(placements, sh, &Placement::normal) != placements.end();
        placements.emplace_back(sh, std::make_pair(sh.bottom(), sh.right()), !duplicate, duplicate);
    }
}

void Piece::cover(coords_t pos, auto &&func) const {
    auto [tgtY, tgtX] = pos;
    for (auto &p : placements) {
        if (!p.enabled)
            continue;
        auto [maxY, maxX] = p.max;
        for (auto [bitY, bitX] : p.normal) {
            if (bitX > tgtX || bitX + maxX < tgtX)
                continue;
            if (bitY > tgtY || bitY + maxY < tgtY)
                continue;
            func(p.normal.translate(tgtX - bitX, tgtY - bitY));
        }
    }
}

Solution::Solution(std::vector<Step> st) : steps{ std::move(st) } {
    for (auto row = 0u; row < Shape::LEN; row++) {
        map.emplace_back();
        for (auto col = 0u; col < Shape::LEN; col++) {
            auto it = std::ranges::find_if(steps, [=](Step st){
                    return st.shape.test(row, col);
            });
            if (it == steps.end())
                map.back().push_back(-1);
            else
                map.back().push_back(std::distance(steps.begin(), it));
        }
    }
}

std::vector<Solution> solve(const std::vector<Piece> &lib, Shape board) {
    std::vector<Solution> solutions;
    std::vector<size_t> used(lib.size(), 0);
    std::vector<Step> history;
    [&](this auto &&self, Shape open_tiles) {
        if (!open_tiles) {
            solutions.emplace_back(history);
            return;
        }
        auto pos = open_tiles.front();
        for (auto &&[p, u, id] : std::views::zip(lib, used, std::views::iota(0zu))) {
            if (u == p.count) continue;
            u++;
            p.cover(pos, [&](Shape placed) {
                if (!(open_tiles >= placed)) return;
                history.push_back(Step{ id, placed });
                self(open_tiles - placed);
                history.pop_back();
            });
            u--;
        }
    }(board);
    return solutions;
}
