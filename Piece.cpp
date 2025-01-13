#include "Piece.hpp"

#include <format>
#include <iostream>

#include <algorithm>
#include <ranges>
#include <stdexcept>

Piece::Piece(Shape s) : count{ 1 }, canonical{ s } {
    for (auto sh : canonical.transforms(true)) {
        if (std::ranges::find(placements, sh, &Placement::normal) != placements.end())
            continue;

        placements.emplace_back(sh, std::make_pair(sh.bottom(), sh.right()));
    }
}

void Piece::cover(coords_t pos, auto &&func) const {
    auto [tgtY, tgtX] = pos;
    for (auto &p : placements) {
        auto [maxY, maxX] = p.max;
        for (auto [bitY, bitX] : p.normal.bits()) {
            if (bitX > tgtX || bitX + maxX < tgtX)
                continue;
            if (bitY > tgtY || bitY + maxY < tgtY)
                continue;
            func(p.normal.translate(tgtX - bitX, tgtY - bitY));
        }
    }
}

Library::Library(std::initializer_list<Shape::shape_t> lst) {
    for (auto sh : lst)
        push(Shape{ sh });
}

// O(n^2) push, won't fix
void Library::push(Shape sh) {
    sh = sh.canonical_form();
    if (auto it = std::ranges::find(lib, sh, &Piece::canonical); it != lib.end()) {
        it->count++;
    } else {
        lib.emplace_back(sh);
    }
}

Solution::Solution(std::vector<Step> history) : steps{ std::move(history) } {
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

std::vector<Solution> Library::solve(Shape board) const {
    std::vector<Solution> solutions;
    std::vector<size_t> used(lib.size(), 0);
    std::vector<Step> history;
    [&,that=this](this auto &&self, Shape open_tiles) {
        if (!open_tiles) {
            solutions.emplace_back(history);
            return;
        }
        auto pos = open_tiles.front();
        history.emplace_back(0u, Shape{ 0u });
        for (auto &&[p, u, id] : std::views::zip(that->lib, used, std::views::iota(0zu))) {
            if (u == p.count) continue;
            u++;
            p.cover(pos, [&](Shape placed) {
                if (!(open_tiles >= placed)) return;
                history.back() = Step{ id, placed };
                self(open_tiles - placed);
            });
            u--;
        }
        history.pop_back();
    }(board);
    return solutions;
}
