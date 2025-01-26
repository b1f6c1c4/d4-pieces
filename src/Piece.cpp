#include "Piece.inl"

#include <algorithm>
#include <limits>
#include <ranges>
#include <stdexcept>

Piece::Piece(Shape s) : count{ 1 }, canonical{ s } {
    for (auto sh : canonical.transforms(true)) {
        auto duplicate = std::ranges::find(placements, sh, &Placement::normal) != placements.end();
        placements.emplace_back(sh, std::make_pair(sh.bottom(), sh.right()), !duplicate, duplicate);
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

Step make_step(size_t id, size_t trs, coords_t tra, Shape placed) {
    int a{}, b{}, c{}, d{};
    switch (trs) {
        case 0: a = +1, d = +1; break;
        case 1: a = -1, d = +1; break;
        case 2: a = +1, d = -1; break;
        case 3: a = -1, d = -1; break;
        case 4: b = +1, c = +1; break;
        case 5: b = -1, c = +1; break;
        case 6: b = +1, c = -1; break;
        case 7: b = -1, c = -1; break;
    }
    return Step{ id, trs, a, b, c, d, tra.second, tra.first, placed };
}

size_t min_tiles(const std::vector<Piece> &lib, const std::vector<size_t> &used) {
    auto m = std::numeric_limits<size_t>::max();
    for (auto &&[p, u] : std::views::zip(lib, used))
        if (u < p.count)
            m = std::min(m, p.canonical.size());
    return m;
}

std::vector<Solution> solve(const std::vector<Piece> &lib, Shape board, bool single) {
    std::vector<Solution> solutions;
    std::vector<size_t> used(lib.size(), 0);
    std::vector<Step> history;
    auto max_tiles = 0zu;
    for (auto &p : lib)
        max_tiles += p.canonical.size() * p.count;
    [&](this auto &&self, Shape open_tiles) {
        if (open_tiles.size() > max_tiles)
            return false;
        if (!open_tiles) {
            solutions.emplace_back(history);
            return single;
        }
        if (open_tiles.size() < min_tiles(lib, used))
            return false;
        auto pos = open_tiles.front();
        for (auto &&[p, u, id] : std::views::zip(lib, used, std::views::iota(0zu))) {
            if (u == p.count) continue;
            u++;
            max_tiles -= p.canonical.size();
            if (p.cover(pos, [&](Shape placed, size_t trs, coords_t tra) {
                if (!(open_tiles >= placed)) return false;
                history.push_back(make_step(id, trs, tra, placed));
                auto f = self(open_tiles - placed);
                history.pop_back();
                return f;
            }))
                return true;
            max_tiles += p.canonical.size();
            u--;
        }
        return false;
    }(board);
    return solutions;
}

size_t solve_count(const std::vector<Piece> &lib, Shape board) {
    auto count = 0zu;
    std::vector<size_t> used(lib.size(), 0);
    auto max_tiles = 0zu;
    for (auto &p : lib)
        max_tiles += p.canonical.size() * p.count;
    [&](this auto &&self, Shape open_tiles) {
        if (open_tiles.size() > max_tiles)
            return;
        if (!open_tiles) {
            count++;
            return;
        }
        if (open_tiles.size() < min_tiles(lib, used))
            return;
        auto pos = open_tiles.front();
        for (auto &&[p, u, id] : std::views::zip(lib, used, std::views::iota(0zu))) {
            if (u == p.count) continue;
            u++;
            max_tiles -= p.canonical.size();
            p.cover(pos, [&](Shape placed, size_t trs, coords_t tra) {
                if (!(open_tiles >= placed)) return false;
                self(open_tiles - placed);
                return false;
            });
            max_tiles += p.canonical.size();
            u--;
        }
    }(board);
    return count;
}
