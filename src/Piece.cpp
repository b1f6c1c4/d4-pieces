#include "Piece.inl"

#include <algorithm>
#include <limits>
#include <ranges>
#include <stdexcept>

#if EMSCRIPTEN
#include <boost/thread/thread.hpp>
#endif

template <size_t L>
Piece<L>::Piece(Shape<L> s, unsigned sym)
    : count{ 1 }, canonical{ s } {
    for (auto sh : canonical.transforms(true)) {
        if (sym & 1u) {
            auto duplicate = std::ranges::find(placements, sh, &Placement::normal) != placements.end();
            placements.emplace_back(sh, std::make_pair(sh.bottom(), sh.right()), !duplicate, duplicate);
        }
        sym >>= 1u;
    }
}

template <size_t L>
Solution<L>::Solution(std::vector<Step<L>> st) : steps{ std::move(st) } {
    for (auto row = 0u; row < Shape<L>::LEN; row++) {
        map.emplace_back();
        for (auto col = 0u; col < Shape<L>::LEN; col++) {
            auto it = std::ranges::find_if(steps, [=](Step<L> st){
                    return st.shape.test(row, col);
            });
            if (it == steps.end())
                map.back().push_back(-1);
            else
                map.back().push_back(std::distance(steps.begin(), it));
        }
    }
}

template <size_t L>
static Step<L> make_step(size_t id, size_t trs, coords_t tra, Shape<L> placed) {
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
    return Step<L>{ id, trs, a, b, c, d, tra.second, tra.first, placed };
}

template <size_t L>
static size_t min_tiles(const std::vector<Piece<L>> &lib, const std::vector<size_t> &used) {
    auto m = std::numeric_limits<size_t>::max();
    for (auto &&[p, u] : std::views::zip(lib, used))
        if (u < p.count)
            m = std::min(m, p.canonical.size());
    return m;
}

template <size_t L>
std::vector<Solution<L>> solve(const std::vector<Piece<L>> &lib, Shape<L> board, bool single) {
    std::vector<Solution<L>> solutions;
    std::vector<size_t> used(lib.size(), 0);
    std::vector<Step<L>> history;
    auto max_tiles = 0zu;
    for (auto &p : lib)
        max_tiles += p.canonical.size() * p.count;
    auto f = [&](auto &&self, Shape<L> open_tiles) {
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
            if (p.cover(pos, [&](Shape<L> placed, size_t trs, coords_t tra) {
                if (!(open_tiles >= placed)) return false;
                history.push_back(make_step(id, trs, tra, placed));
                auto f = self(self, open_tiles - placed);
                history.pop_back();
                return f;
            }))
                return true;
            max_tiles += p.canonical.size();
            u--;
        }
        return false;
    };
    f(f, board);
    return solutions;
}

template <size_t L>
size_t solve_count(const std::vector<Piece<L>> &lib, Shape<L> board) {
    auto count = 0zu;
    std::vector<size_t> used(lib.size(), 0);
    auto max_tiles = 0zu;
    for (auto &p : lib)
        max_tiles += p.canonical.size() * p.count;
    auto f = [&](auto &&self, Shape<L> open_tiles) {
#if EMSCRIPTEN
        boost::this_thread::interruption_point();
#endif
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
            p.cover(pos, [&](Shape<L> placed, size_t trs, coords_t tra) {
                if (!(open_tiles >= placed)) return false;
                self(self, open_tiles - placed);
                return false;
            });
            max_tiles += p.canonical.size();
            u--;
        }
    };
    f(f, board);
    return count;
}

template Piece<8>::Piece(Shape<8> s, unsigned sym);
template Solution<8>::Solution(std::vector<Step<8>> st);
template std::vector<Solution<8>> solve(const std::vector<Piece<8>> &lib, Shape<8> board, bool single);
template size_t solve_count(const std::vector<Piece<8>> &lib, Shape<8> board);

template Piece<11>::Piece(Shape<11> s, unsigned sym);
template Solution<11>::Solution(std::vector<Step<11>> st);
template std::vector<Solution<11>> solve(const std::vector<Piece<11>> &lib, Shape<11> board, bool single);
template size_t solve_count(const std::vector<Piece<11>> &lib, Shape<11> board);
